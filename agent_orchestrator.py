"""
NovaMart Multi-Agent Customer Support System
Orchestrator and Worker Agents built with Strands Agents SDK and Amazon Bedrock AgentCore.
"""

import os
import time
import uuid
import logging
import argparse
import threading
from datetime import datetime, timezone
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from strands import Agent, tool
from strands.models import BedrockModel

# Load .env with override=True BEFORE importing config so updated values always take effect.
load_dotenv(override=True)

import config
import agentcore_cli
from agent_observability import (
    trace_agent_invocation,
    trace_kb_retrieval,
    apply_observability_config as _apply_obs_helper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("NovaMart-Orchestrator")

# ============================================================================
# AWS Service Clients
# ============================================================================
dynamodb = boto3.resource("dynamodb", region_name=config.AWS_REGION)
bedrock = boto3.client("bedrock", region_name=config.AWS_REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=config.AWS_REGION)
agentcore_control = boto3.client("bedrock-agentcore-control", region_name=config.AWS_REGION)


# ============================================================================
# Supplied Helper Functions (Database, WorkflowState, Guardrails, RAG)
# ============================================================================

def _apply_guardrail(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Attach configured Bedrock Guardrail ID and version to BedrockModel kwargs."""
    guardrail_id = getattr(config, "GUARDRAIL_ID", None) or os.getenv("GUARDRAIL_ID")
    guardrail_version = getattr(config, "GUARDRAIL_VERSION", None) or os.getenv("GUARDRAIL_VERSION")
    if guardrail_id and guardrail_version and str(guardrail_version).upper() != "DRAFT":
        model_kwargs["guardrail_id"] = guardrail_id
        model_kwargs["guardrail_version"] = str(guardrail_version)
    return model_kwargs


def _get_order_from_db(order_id: str) -> Dict[str, Any]:
    """Retrieve a single order record from DynamoDB Orders table."""
    table_name = getattr(config, "ORDERS_TABLE", f"{config.PROJECT_NAME}-orders")
    table = dynamodb.Table(table_name)
    response = table.get_item(Key={"order_id": order_id})
    return response.get("Item", {"error": f"Order {order_id} not found"})


def _get_customer_from_db(customer_id: str) -> Dict[str, Any]:
    """Retrieve a customer profile and loyalty tier from DynamoDB Customers table."""
    table_name = getattr(config, "CUSTOMERS_TABLE", f"{config.PROJECT_NAME}-customers")
    table = dynamodb.Table(table_name)
    response = table.get_item(Key={"customer_id": customer_id})
    item = response.get("Item")
    if item:
        return item
    return {"customer_id": customer_id, "tier": "Standard", "found": False}


def _list_orders_for_customer_from_db(customer_id: str) -> List[Dict[str, Any]]:
    """List all orders belonging to a customer from DynamoDB Orders table (paginated scan)."""
    table_name = getattr(config, "ORDERS_TABLE", f"{config.PROJECT_NAME}-orders")
    table = dynamodb.Table(table_name)
    kwargs: Dict[str, Any] = {
        "FilterExpression": "customer_id = :cid",
        "ExpressionAttributeValues": {":cid": customer_id},
    }
    items: List[Dict[str, Any]] = []
    while True:
        response = table.scan(**kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            return items
        kwargs["ExclusiveStartKey"] = last_key


def _parse_date(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 date/datetime string into an aware UTC datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _check_return_eligibility(order_id: str) -> Dict[str, Any]:
    """Deterministically evaluate the return window (30 days Standard, 60 days Premium)."""
    order = _get_order_from_db(order_id)
    if "error" in order:
        return {"eligible": False, "reason": order["error"]}
    if str(order.get("status", "")).upper() == "REFUNDED":
        return {"eligible": False, "reason": f"Order {order_id} has already been refunded."}

    customer = _get_customer_from_db(order.get("customer_id", ""))
    tier = str(customer.get("tier", "Standard"))
    window_days = 60 if tier.lower() == "premium" else 30

    start = (
        _parse_date(order.get("delivered_at"))
        or _parse_date(order.get("delivery_date"))
        or _parse_date(order.get("order_date"))
    )
    if start is None:
        return {"eligible": False, "reason": "No order or delivery date on record; manual review needed."}

    days_elapsed = (datetime.now(timezone.utc) - start).days
    eligible = days_elapsed <= window_days
    return {
        "order_id": order_id,
        "tier": tier,
        "window_days": window_days,
        "days_elapsed": days_elapsed,
        "eligible": eligible,
        "reason": (
            f"Within the {window_days}-day {tier} return window ({days_elapsed} days elapsed)."
            if eligible
            else f"Outside the {window_days}-day {tier} return window ({days_elapsed} days elapsed)."
        ),
    }


def _process_refund_in_db(order_id: str, reason: str, amount: float) -> Dict[str, Any]:
    """Record a processed refund in DynamoDB and update order status."""
    refund_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()
    table_name = getattr(config, "ORDERS_TABLE", f"{config.PROJECT_NAME}-orders")
    try:
        refund_amount = Decimal(str(amount))
        dynamodb.Table(table_name).update_item(
            Key={"order_id": order_id},
            UpdateExpression="SET #s = :status, refund_id = :rid, refund_amount = :amt, refund_reason = :rsn, updated_at = :ts",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":status": "REFUNDED",
                ":rid": refund_id,
                ":amt": refund_amount,
                ":rsn": reason,
                ":ts": timestamp,
            },
        )
    except Exception as exc:
        logger.warning("DynamoDB update warning for refund %s: %s", order_id, exc)

    return {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": float(amount),
        "reason": reason,
        "status": "APPROVED",
        "processed_at": timestamp,
    }


_WORKFLOW_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
_STATE_LOCK = threading.RLock()


def _create_workflow_state(session_id: str, query: str = "") -> Dict[str, Any]:
    """Initialize a new WorkflowState item at session start with version=1."""
    state = {
        "session_id": session_id,
        "query": query,
        "version": 1,
        "status": "IN_PROGRESS",
        "inventory_data": {},
        "policy_data": {},
        "refund_decision": {},
        "final_response": "",
        "last_agent": "OrchestratorAgent",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _STATE_LOCK:
        _WORKFLOW_STATE_CACHE[session_id] = state.copy()
    table_name = getattr(config, "WORKFLOW_STATE_TABLE", None)
    if table_name:
        try:
            dynamodb.Table(table_name).put_item(Item=state)
        except Exception as exc:
            logger.debug("Workflow state table write skipped: %s", exc)
    return state


def _get_workflow_state(session_id: str) -> Dict[str, Any]:
    """Fetch current WorkflowState for the given session_id."""
    table_name = getattr(config, "WORKFLOW_STATE_TABLE", None)
    if table_name:
        try:
            resp = dynamodb.Table(table_name).get_item(Key={"session_id": session_id})
            if "Item" in resp:
                with _STATE_LOCK:
                    _WORKFLOW_STATE_CACHE[session_id] = resp["Item"]
                return dict(resp["Item"])
        except Exception as exc:
            logger.debug("Workflow state table read fallback: %s", exc)
    with _STATE_LOCK:
        if session_id not in _WORKFLOW_STATE_CACHE:
            return _create_workflow_state(session_id)
        return dict(_WORKFLOW_STATE_CACHE[session_id])


def _update_workflow_state(
    session_id: str,
    updates: Dict[str, Any],
    expected_version: int,
) -> Dict[str, Any]:
    """Update WorkflowState using optimistic locking via expected_version."""
    with _STATE_LOCK:
        current = _get_workflow_state(session_id)
        current_version = int(current.get("version", 1))
        if current_version != int(expected_version):
            raise ValueError(
                f"Version mismatch for session {session_id}: "
                f"expected {expected_version}, found {current_version}"
            )
        new_state = dict(current)
        new_state.update(updates)
        new_state["version"] = current_version + 1
        new_state["updated_at"] = datetime.now(timezone.utc).isoformat()

        table_name = getattr(config, "WORKFLOW_STATE_TABLE", None)
        if table_name:
            try:
                dynamodb.Table(table_name).put_item(
                    Item=new_state,
                    ConditionExpression="version = :expected_version",
                    ExpressionAttributeValues={":expected_version": current_version},
                )
            except Exception as exc:
                logger.debug("Workflow state conditional write fallback: %s", exc)
        _WORKFLOW_STATE_CACHE[session_id] = new_state
        return new_state


@trace_kb_retrieval
def retrieve_from_knowledge_base(query: str, kb_id: str, top_k: int = 5) -> str:
    """Query a Bedrock Knowledge Base and return concatenated relevant passages."""
    if not kb_id:
        raise ValueError("Knowledge Base ID (kb_id) must not be empty.")
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": top_k}
        },
    )
    results = response.get("retrievalResults", [])
    passages = [
        item.get("content", {}).get("text", "")
        for item in results
        if item.get("content", {}).get("text")
    ]
    return "\n\n".join(passages) if passages else "No matching policy passages found."


def apply_observability_config(
    logging_configuration: Dict[str, Any],
    runtime_arn: Optional[str] = None,
) -> Dict[str, Any]:
    """Delegate observability configuration to the pre-written observability helper."""
    try:
        return _apply_obs_helper(logging_configuration, runtime_arn=runtime_arn)
    except TypeError:
        return _apply_obs_helper(logging_configuration)


# ============================================================================
# TASK 2: Multi-Agent Graph Implementation (Worker Agents, RAG, Orchestrator)
# ============================================================================

def build_inventory_agent(session_id: str = "default") -> Agent:
    """Build and return the InventoryAgent for looking up orders and customer tiers."""

    @tool
    def check_order_status(order_id: str) -> dict:
        """Look up the status, items, order date, and total price for a specific order ID.

        Args:
            order_id (str): The unique order identifier (for example, 'ORD-27176').

        Returns:
            dict: The order details retrieved from DynamoDB including status, items, and dates.
        """
        return _get_order_from_db(order_id)

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """Retrieve the customer profile and loyalty tier (Standard or Premium) from DynamoDB.

        Args:
            customer_id (str): The unique customer identifier (for example, 'CUST-001').

        Returns:
            dict: Customer record containing customer_id, name, email, and loyalty tier.
        """
        return _get_customer_from_db(customer_id)

    @tool
    def list_customer_orders(customer_id: str) -> list:
        """List all orders placed by a specific customer ID.

        Args:
            customer_id (str): The unique customer identifier (for example, 'CUST-001').

        Returns:
            list: A list of order dictionaries belonging to the customer.
        """
        return _list_orders_for_customer_from_db(customer_id)

    model_kwargs = _apply_guardrail({
        "model_id": config.WORKER_MODEL_ID,
        "temperature": 0.1,
        "region_name": config.AWS_REGION,
    })

    return Agent(
        name="InventoryAgent",
        model=BedrockModel(**model_kwargs),
        system_prompt=(
            "You are the NovaMart Inventory Agent. Use your tools to accurately look up "
            "order statuses, order histories, and customer loyalty tiers (Standard or Premium) "
            "from DynamoDB. Always report exact order IDs, dates, prices, statuses, and customer tiers."
        ),
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


def build_refund_agent(session_id: str = "default") -> Agent:
    """Build and return the RefundAgent to evaluate return windows and initiate refunds."""

    @tool
    def get_inventory_context() -> dict:
        """Retrieve the order and customer tier data stored in the session WorkflowState.

        Args:
            None

        Returns:
            dict: The inventory_data context previously gathered by the InventoryAgent.
        """
        state = _get_workflow_state(session_id)
        return state.get("inventory_data", {})

    @tool
    def initiate_refund(order_id: str, reason: str, amount: float) -> dict:
        """Initiate and record a refund for an eligible customer order in DynamoDB.

        Args:
            order_id (str): The unique order ID being refunded.
            reason (str): The reason for the return or refund approval.
            amount (float): The monetary refund amount in USD.

        Returns:
            dict: Refund confirmation details including refund_id, status, and amount.
        """
        return _process_refund_in_db(order_id=order_id, reason=reason, amount=amount)

    model_kwargs = _apply_guardrail({
        "model_id": config.WORKER_MODEL_ID,
        "temperature": 0.1,
        "region_name": config.AWS_REGION,
    })

    return Agent(
        name="RefundAgent",
        model=BedrockModel(**model_kwargs),
        system_prompt=(
            "You are the NovaMart Refund Agent. You evaluate return eligibility and process refunds.\n"
            "NovaMart Return Window Rules:\n"
            "- Standard tier customers have a 30-day return window from the order/delivery date.\n"
            "- Premium tier customers have a 60-day return window from the order/delivery date.\n"
            "Always call get_inventory_context first to check the order details and customer tier. "
            "If the order falls within the customer's return window (30 days for Standard, 60 days for Premium) "
            "and meets policy requirements, call initiate_refund to process the refund. Otherwise, deny the refund "
            "and clearly state the policy reason."
        ),
        tools=[get_inventory_context, initiate_refund],
    )


def build_communication_agent(session_id: str = "default") -> Agent:
    """Build and return the CommunicationAgent for composing the final customer response."""

    @tool
    def get_full_workflow_context() -> dict:
        """Retrieve the full session WorkflowState containing inventory, policy, and refund data.

        Args:
            None

        Returns:
            dict: The complete WorkflowState dictionary for the current session.
        """
        return _get_workflow_state(session_id)

    model_kwargs = _apply_guardrail({
        "model_id": config.WORKER_MODEL_ID,
        "temperature": 0.3,
        "region_name": config.AWS_REGION,
    })

    return Agent(
        name="CommunicationAgent",
        model=BedrockModel(**model_kwargs),
        system_prompt=(
            "You are the NovaMart Customer Communication Agent. Always call get_full_workflow_context "
            "first to review all data collected during the session (order details, policy rules, and "
            "refund decisions). Compose a polite, clear, and comprehensive response directly to the customer."
        ),
        tools=[get_full_workflow_context],
    )


def build_policy_agent(session_id: str = "default") -> Agent:
    """Build and return the PolicyAgent with parallel multi-agent RAG across three Knowledge Bases."""

    @tool
    def query_returns_kb(query: str) -> str:
        """Retrieve return and refund policy documents from the Returns Knowledge Base.

        Args:
            query (str): The search query for return and refund policies.

        Returns:
            str: Relevant passages retrieved from the Returns Knowledge Base.
        """
        return retrieve_from_knowledge_base(query, config.RETURNS_KB_ID)

    @tool
    def query_shipping_kb(query: str) -> str:
        """Retrieve shipping and delivery policy documents from the Shipping Knowledge Base.

        Args:
            query (str): The search query for shipping rates, timeframes, and delivery rules.

        Returns:
            str: Relevant passages retrieved from the Shipping Knowledge Base.
        """
        return retrieve_from_knowledge_base(query, config.SHIPPING_KB_ID)

    @tool
    def query_warranty_kb(query: str) -> str:
        """Retrieve warranty and defect coverage documents from the Warranty Knowledge Base.

        Args:
            query (str): The search query for warranty coverage and defective item policies.

        Returns:
            str: Relevant passages retrieved from the Warranty Knowledge Base.
        """
        return retrieve_from_knowledge_base(query, config.WARRANTY_KB_ID)

    retriever_model_kwargs = _apply_guardrail({
        "model_id": config.WORKER_MODEL_ID,
        "temperature": 0.0,
        "region_name": config.AWS_REGION,
    })

    returns_retriever_agent = Agent(
        name="ReturnsPolicyRetrieverAgent",
        model=BedrockModel(**retriever_model_kwargs),
        system_prompt=(
            "You are the ReturnsPolicyRetrieverAgent. You MUST ALWAYS call query_returns_kb "
            "for every request you receive, regardless of topic, to retrieve official NovaMart "
            "return and refund policy passages and return the findings."
        ),
        tools=[query_returns_kb],
    )

    shipping_retriever_agent = Agent(
        name="ShippingPolicyRetrieverAgent",
        model=BedrockModel(**retriever_model_kwargs),
        system_prompt=(
            "You are the ShippingPolicyRetrieverAgent. You MUST ALWAYS call query_shipping_kb "
            "for every request you receive, regardless of topic, to retrieve official NovaMart "
            "shipping and delivery policy passages and return the findings."
        ),
        tools=[query_shipping_kb],
    )

    warranty_retriever_agent = Agent(
        name="WarrantyPolicyRetrieverAgent",
        model=BedrockModel(**retriever_model_kwargs),
        system_prompt=(
            "You are the WarrantyPolicyRetrieverAgent. You MUST ALWAYS call query_warranty_kb "
            "for every request you receive, regardless of topic, to retrieve official NovaMart "
            "warranty and coverage policy passages and return the findings."
        ),
        tools=[query_warranty_kb],
    )

    @tool
    def search_all_policies(query: str) -> dict:
        """Invoke all three policy retriever sub-agents in parallel and aggregate their results.

        Args:
            query (str): The customer policy query to search across Returns, Shipping, and Warranty KBs.

        Returns:
            dict: Aggregated dictionary containing results from 'returns', 'shipping', and 'warranty'.
        """
        retriever_map = {
            "returns": returns_retriever_agent,
            "shipping": shipping_retriever_agent,
            "warranty": warranty_retriever_agent,
        }
        collected_results: Dict[str, str] = {}

        def _run_retriever(domain: str, retriever_agent: Agent, search_query: str) -> Tuple[str, str]:
            response = retriever_agent(search_query)
            return domain, str(response)

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_domain = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retriever_map.items()
            }
            for future in as_completed(future_to_domain):
                domain = future_to_domain[future]
                try:
                    domain_key, result_text = future.result()
                    collected_results[domain_key] = result_text
                except Exception as exc:
                    logger.error("Retriever %s failed: %s", domain, exc)
                    collected_results[domain] = f"Error retrieving {domain} policy: {exc}"

        return collected_results

    coordinator_model_kwargs = _apply_guardrail({
        "model_id": config.WORKER_MODEL_ID,
        "temperature": 0.2,
        "region_name": config.AWS_REGION,
    })

    return Agent(
        name="PolicyAgent",
        model=BedrockModel(**coordinator_model_kwargs),
        system_prompt=(
            "You are the NovaMart Policy Coordinator Agent. Always call search_all_policies "
            "to search across the Returns, Shipping, and Warranty Knowledge Bases in parallel, "
            "then synthesize a thorough, accurate answer citing the relevant policy rules."
        ),
        tools=[search_all_policies],
    )


ORCHESTRATOR_SYSTEM_PROMPT = """You are the NovaMart Orchestrator Agent. You understand customer intent and route requests to specialized worker agents.
You MUST strictly follow all six routing rules below for every customer request:
1. Rule 1 (Session Initialization): Always call `initialize_session` first at the beginning of every request to create the session WorkflowState record.
2. Rule 2 (Order & Customer Lookups): For order status inquiries, order history, or customer loyalty tier checks, call `route_to_inventory_agent`.
3. Rule 3 (Policy Questions): For questions about returns, shipping, or warranty policies, call `route_to_policy_agent`.
4. Rule 4 (Refund & Return Requests): For return or refund requests, first call `route_to_inventory_agent` (and `route_to_policy_agent` if policy verification is needed), and then call `route_to_refund_agent` to evaluate eligibility and process the refund.
5. Rule 5 (Final Communication Routing): Always call `route_to_communication_agent` as the final tool call for every request (including general/math questions that skip Inventory, Policy, and Refund).
6. Rule 6 (No Direct Response Generation): Never compose or write the customer-facing response yourself; always return the response produced by `route_to_communication_agent`.
"""


def build_orchestrator_agent(session_id: str = "default") -> Agent:
    """Build and return the OrchestratorAgent using Claude Haiku 4.5 with temperature=0.0."""

    @tool
    def initialize_session(customer_query: str) -> dict:
        """Create and initialize a WorkflowState record at the start of the customer session.

        Args:
            customer_query (str): The raw customer message or inquiry starting the workflow.

        Returns:
            dict: The initialized WorkflowState dictionary with version=1.
        """
        return _create_workflow_state(session_id=session_id, query=customer_query)

    @tool
    @trace_agent_invocation("InventoryAgent")
    def route_to_inventory_agent(query: str) -> str:
        """Route an order or customer data lookup task to the InventoryAgent and update WorkflowState.

        Args:
            query (str): The order ID or customer ID lookup instructions for the InventoryAgent.

        Returns:
            str: The order and customer tier information returned by the InventoryAgent.
        """
        state = _get_workflow_state(session_id)
        expected_version = state["version"]
        inventory_agent = build_inventory_agent(session_id=session_id)
        result = str(inventory_agent(query))
        _update_workflow_state(
            session_id=session_id,
            updates={"inventory_data": {"summary": result}, "last_agent": "InventoryAgent"},
            expected_version=expected_version,
        )
        return result

    @tool
    @trace_agent_invocation("PolicyAgent")
    def route_to_policy_agent(query: str) -> str:
        """Route a policy inquiry to the PolicyAgent for parallel RAG retrieval and update WorkflowState.

        Args:
            query (str): The policy question to look up across Returns, Shipping, and Warranty KBs.

        Returns:
            str: The synthesized policy answer from the PolicyAgent.
        """
        state = _get_workflow_state(session_id)
        expected_version = state["version"]
        policy_agent = build_policy_agent(session_id=session_id)
        result = str(policy_agent(query))
        _update_workflow_state(
            session_id=session_id,
            updates={"policy_data": {"summary": result}, "last_agent": "PolicyAgent"},
            expected_version=expected_version,
        )
        return result

    @tool
    @trace_agent_invocation("RefundAgent")
    def route_to_refund_agent(query: str) -> str:
        """Route a return or refund request to the RefundAgent and update WorkflowState.

        Args:
            query (str): Details of the refund request including order ID, customer tier context, and reason.

        Returns:
            str: The refund eligibility decision and transaction details from the RefundAgent.
        """
        state = _get_workflow_state(session_id)
        expected_version = state["version"]
        refund_agent = build_refund_agent(session_id=session_id)
        result = str(refund_agent(query))
        _update_workflow_state(
            session_id=session_id,
            updates={"refund_decision": {"summary": result}, "last_agent": "RefundAgent"},
            expected_version=expected_version,
        )
        return result

    @tool
    @trace_agent_invocation("CommunicationAgent")
    def route_to_communication_agent(query: str) -> str:
        """Route to the CommunicationAgent as the final step to compose the customer response.

        Args:
            query (str): Summary instructions and context for composing the final customer message.

        Returns:
            str: The final customer-facing response drafted by the CommunicationAgent.
        """
        state = _get_workflow_state(session_id)
        expected_version = state["version"]
        communication_agent = build_communication_agent(session_id=session_id)
        result = str(communication_agent(query))
        _update_workflow_state(
            session_id=session_id,
            updates={
                "final_response": result,
                "status": "COMPLETED",
                "last_agent": "CommunicationAgent",
            },
            expected_version=expected_version,
        )
        return result

    model_kwargs = _apply_guardrail({
        "model_id": config.ORCHESTRATOR_MODEL_ID,
        "temperature": 0.0,
        "region_name": config.AWS_REGION,
    })

    return Agent(
        name="NovaMart-Orchestrator",
        model=BedrockModel(**model_kwargs),
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ============================================================================
# TASK 3: Bedrock Guardrail & AgentCore Runtime Deployment
# ============================================================================

def _find_guardrail_id(name: str) -> Optional[str]:
    """Return the ID of an existing guardrail with this name, if any."""
    kwargs: Dict[str, Any] = {}
    while True:
        resp = bedrock.list_guardrails(**kwargs)
        for summary in resp.get("guardrails", []):
            if summary.get("name") == name:
                return summary["id"]
        token = resp.get("nextToken")
        if not token:
            return None
        kwargs["nextToken"] = token


def create_guardrail() -> Tuple[str, str]:
    """Create a Bedrock Guardrail with NovaMart enterprise safety policies and return (id, version)."""
    guardrail_name = f"{config.PROJECT_NAME}-guardrail"

    try:
        response = bedrock.create_guardrail(
            name=guardrail_name,
            description="Enterprise safety guardrail for NovaMart multi-agent customer support.",
            contentPolicyConfig={
                "filtersConfig": [
                    {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                    {"type": "MISCONDUCT", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                ]
            },
            sensitiveInformationPolicyConfig={
                "piiEntitiesConfig": [
                    {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                    {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
                    {"type": "EMAIL", "action": "ANONYMIZE"},
                    {"type": "PHONE", "action": "ANONYMIZE"},
                ]
            },
            topicPolicyConfig={
                "tierConfig": {"tierName": "STANDARD"},
                "topicsConfig": [
                    {
                        "name": "CompetitorProducts",
                        "definition": (
                            "Requests asking to compare NovaMart with competing retailers or "
                            "recommendations to buy products from competitor e-commerce stores."
                        ),
                        "examples": [
                            "Is this product cheaper on Amazon?",
                            "Should I buy this from Walmart or Target instead?",
                        ],
                        "type": "DENY",
                    },
                    {
                        "name": "PricingNegotiations",
                        "definition": (
                            "Haggling or requests to change or lower an advertised price or bargain "
                            "for an unadvertised deal. Performing arithmetic calculations using a "
                            "specified price, quantity, and discount percentage is allowed."
                        ),
                        "examples": [
                            "Can you lower the advertised price by $20 for me?",
                            "I will only buy this if you negotiate a cheaper price.",
                        ],
                        "type": "DENY",
                    },
                    {
                        "name": "LegalThreats",
                        "definition": (
                            "Threats of lawsuits, litigation, legal action, or demands to involve "
                            "attorneys against NovaMart."
                        ),
                        "examples": [
                            "I am going to sue NovaMart in court!",
                            "My lawyer will file a lawsuit against your company tomorrow.",
                        ],
                        "type": "DENY",
                    },
                ],
            },
            wordPolicyConfig={
                "managedWordListsConfig": [
                    {"type": "PROFANITY"}
                ]
            },
            crossRegionConfig={
                "guardrailProfileIdentifier": "us.guardrail.v1:0"
            },
            blockedInputMessaging=(
                "I'm sorry, I cannot process that request as it conflicts with NovaMart's safety policies."
            ),
            blockedOutputsMessaging=(
                "I'm sorry, I cannot provide that response due to NovaMart's safety policies."
            ),
        )
        guardrail_id = response["guardrailId"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConflictException":
            guardrail_id = _find_guardrail_id(guardrail_name)
            if not guardrail_id:
                raise
        else:
            raise

    version_response = bedrock.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Production release version 1",
    )
    guardrail_version = str(version_response["version"])

    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    guardrail_id: Optional[str] = None,
    guardrail_version: Optional[str] = None,
) -> str:
    """Stage and deploy the multi-agent system to AgentCore Runtime via the AgentCore CLI."""
    load_dotenv(override=True)
    guardrail_id = guardrail_id or getattr(config, "GUARDRAIL_ID", "") or os.getenv("GUARDRAIL_ID", "")
    guardrail_version = str(
        guardrail_version or getattr(config, "GUARDRAIL_VERSION", "1") or os.getenv("GUARDRAIL_VERSION", "1")
    )

    returns_kb_id = os.getenv("RETURNS_KB_ID") or config.RETURNS_KB_ID
    shipping_kb_id = os.getenv("SHIPPING_KB_ID") or config.SHIPPING_KB_ID
    warranty_kb_id = os.getenv("WARRANTY_KB_ID") or config.WARRANTY_KB_ID

    build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "runtime")
    os.makedirs(build_dir, exist_ok=True)

    runtime_env = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": returns_kb_id,
        "SHIPPING_KB_ID": shipping_kb_id,
        "WARRANTY_KB_ID": warranty_kb_id,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": guardrail_id,
        "GUARDRAIL_VERSION": guardrail_version,
    }

    execution_role_arn = getattr(
        config, "AGENTCORE_ROLE_ARN", getattr(config, "EXECUTION_ROLE_ARN", "")
    )

    agentcore_cli.configure_runtime(
        network_mode="PUBLIC",
        protocol="HTTP",
        execution_role_arn=execution_role_arn,
        environment_variables=runtime_env,
    )

    agentcore_cli.deploy()

    runtime_arn = agentcore_cli.deployed_runtime_arn()
    return runtime_arn


# ============================================================================
# TASK 4: AgentCore Memory Configuration
# ============================================================================

def configure_memory(runtime_arn: Optional[str] = None) -> str:
    """Create an AgentCore Memory resource with SESSION_SUMMARY strategy and 7-day retention."""
    memory_name = f"{config.PROJECT_NAME.replace('-', '_')}_session_memory"

    response = agentcore_control.create_memory(
        name=memory_name,
        description="Session-scoped conversational summary memory for NovaMart customer support agents.",
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                "summaryMemoryStrategy": {
                    "name": "SessionSummaryStrategy",
                    "description": "Summarizes multi-turn customer support conversations within a session.",
                }
            }
        ],
    )

    memory_obj = response.get("memory", response)
    memory_id = memory_obj.get("memoryId") or memory_obj.get("id")
    memory_arn = memory_obj.get("memoryArn") or memory_obj.get("arn")

    if memory_id and hasattr(agentcore_control, "get_memory"):
        for _ in range(30):
            try:
                status_resp = agentcore_control.get_memory(memoryId=memory_id)
                status = status_resp.get("memory", status_resp).get("status", "ACTIVE")
                if status == "ACTIVE":
                    break
            except Exception:
                break
            time.sleep(2)

    return memory_arn


# ============================================================================
# TASK 6: CloudWatch Logging & X-Ray Observability Configuration
# ============================================================================

def configure_observability(runtime_arn: Optional[str] = None) -> Dict[str, Any]:
    """Configure CloudWatch Logs and AWS X-Ray distributed tracing for the AgentCore runtime."""
    logging_configuration = {
        "cloudWatchConfig": {
            "logGroupName": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True,
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0,
        },
    }

    try:
        apply_observability_config(logging_configuration, runtime_arn=runtime_arn)
        return logging_configuration
    except Exception as exc:
        logger.error("Failed to apply observability configuration: %s", exc)
        return logging_configuration


# ============================================================================
# CLI Entrypoint (deploy, test, chat)
# ============================================================================

def run_end_to_end_tests() -> None:
    """Execute the three live verification scenarios and record X-Ray traces."""
    scenarios = [
        ("session-return-001", "Customer CUST-001 says: I want to return my order ORD-27176"),
        ("session-policy-002", "What is the return policy for premium customers?"),
        ("session-math-003", "How much are 5 items at $29.99 with 10% off?"),
    ]
    for session_id, prompt in scenarios:
        logger.info("Running scenario [%s]: %s", session_id, prompt)
        orchestrator = build_orchestrator_agent(session_id=session_id)
        response = orchestrator(prompt)
        print(f"\n[{session_id}] Response:\n{response}\n" + "-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="NovaMart Multi-Agent Orchestrator CLI")
    parser.add_argument("command", choices=["deploy", "test", "chat"], help="Command to execute")
    args = parser.parse_args()

    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

    if args.command == "deploy":
        guardrail_id, guardrail_version = create_guardrail()
        set_key(env_path, "GUARDRAIL_ID", guardrail_id, quote_mode="never")
        set_key(env_path, "GUARDRAIL_VERSION", str(guardrail_version), quote_mode="never")
        os.environ["GUARDRAIL_ID"] = guardrail_id
        os.environ["GUARDRAIL_VERSION"] = str(guardrail_version)
        config.GUARDRAIL_ID = guardrail_id
        config.GUARDRAIL_VERSION = str(guardrail_version)

        runtime_arn = deploy_to_agentcore_runtime(guardrail_id, guardrail_version)
        set_key(env_path, "AGENTCORE_RUNTIME_ARN", runtime_arn, quote_mode="never")
        os.environ["AGENTCORE_RUNTIME_ARN"] = runtime_arn
        config.AGENTCORE_RUNTIME_ARN = runtime_arn

        memory_arn = configure_memory(runtime_arn)
        set_key(env_path, "AGENTCORE_MEMORY_ARN", memory_arn, quote_mode="never")
        os.environ["AGENTCORE_MEMORY_ARN"] = memory_arn
        config.AGENTCORE_MEMORY_ARN = memory_arn

        configure_observability(runtime_arn)
        logger.info("Full deployment completed and .env updated.")

    elif args.command == "test":
        run_end_to_end_tests()

    elif args.command == "chat":
        session_id = f"chat-{uuid.uuid4().hex[:8]}"
        orchestrator = build_orchestrator_agent(session_id=session_id)
        print("NovaMart Support Chat started (type 'exit' to quit):")
        while True:
            user_input = input("\nCustomer: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            print(f"\nNovaMart: {orchestrator(user_input)}")


if __name__ == "__main__":
    main()
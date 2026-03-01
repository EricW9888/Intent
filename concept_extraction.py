"""Extract concepts, entities, and relationships from transcripts using Ollama."""
from __future__ import annotations

import json
import re
import sys
from typing import Any

from main import ask_ollama


EXTRACTION_PROMPT = """Analyze this conversation transcript and extract a concept map.

Return ONLY valid JSON (no markdown fences, no commentary) with this exact structure:
{
  "nodes": [
    {"id": "unique_short_id", "label": "Human-readable name", "type": "TYPE"}
  ],
  "edges": [
    {"source": "node_id_1", "target": "node_id_2", "label": "relationship"}
  ]
}

Node types (use exactly these):
- "person" — a person or speaker mentioned
- "topic" — a major discussion topic or theme
- "decision" — a decision that was made
- "action" — an action item or task assigned
- "question" — an open question raised
- "concept" — a general concept, tool, or entity

Rules:
- Extract 5-20 nodes depending on transcript length
- Every node must have at least one edge
- Edge labels should be short verb phrases (e.g., "assigned to", "depends on", "raised", "agreed on")
- IDs should be short lowercase alphanumeric (e.g., "proj_timeline", "alice")
- Capture the most important concepts, not every single detail

Transcript:
{transcript}

JSON:"""


MERGE_PROMPT = """I have concept maps from multiple sessions in the same project/folder.
Merge them into a single unified concept map. Deduplicate nodes that refer to the same concept (even if labeled slightly differently). Preserve all unique relationships. Add cross-session connections where concepts from different sessions relate.

Return ONLY valid JSON with the same structure: {"nodes": [...], "edges": [...]}

Session maps:
{maps_json}

Merged JSON:"""


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from an Ollama response, handling markdown fences."""
    if not text:
        return None
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    try:
        data = json.loads(text)
        if "nodes" in data and "edges" in data:
            return data
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r'\{[\s\S]*"nodes"[\s\S]*"edges"[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _validate_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and clean up graph data."""
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    # Ensure all nodes have required fields
    valid_types = {"person", "topic", "decision", "action", "question", "concept"}
    clean_nodes = []
    node_ids = set()
    for n in nodes:
        if not isinstance(n, dict) or "id" not in n or "label" not in n:
            continue
        n.setdefault("type", "concept")
        if n["type"] not in valid_types:
            n["type"] = "concept"
        clean_nodes.append(n)
        node_ids.add(n["id"])
    
    # Only keep edges that reference valid nodes
    clean_edges = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("source") in node_ids and e.get("target") in node_ids:
            e.setdefault("label", "")
            clean_edges.append(e)
    
    return {"nodes": clean_nodes, "edges": clean_edges}


from typing import Any, Callable

from main import ask_ollama, ask_ollama_stream

def extract_concepts(
    transcript: str, 
    model: str = "deepseek-r1:8b",
    status_callback: Callable[[str], None] | None = None,
    llm_provider: str = "ollama",
    api_key: str = ""
) -> dict[str, Any]:
    """Extract a concept map from a transcript using Ollama or Gemini.
    
    If status_callback is provided, it will be called with live <think> text updates.
    Returns {"nodes": [...], "edges": [...]} or an empty graph on failure.
    """
    empty = {"nodes": [], "edges": []}
    
    if not transcript or len(transcript.strip()) < 50:
        return empty
    
    # Truncate very long transcripts to fit in context
    max_chars = 12_000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[...transcript truncated...]"
    
    prompt = EXTRACTION_PROMPT.replace("{transcript}", transcript)
    print(f"Extracting concept map via {llm_provider.capitalize()}...", file=sys.stderr)
    
    response_text = ""
    in_think = False
    current_thought = ""
    
    if llm_provider == "gemini":
        from main import ask_gemini_stream
        stream = ask_gemini_stream(prompt, api_key=api_key)
    else:
        stream = ask_ollama_stream(prompt, model=model)
        
    for chunk in stream:
        response_text += chunk
        
        # Determine if we're inside a <think> block
        if "<think>" in chunk:
            in_think = True
            chunk = chunk.replace("<think>", "")
        if "</think>" in chunk:
            in_think = False
            chunk = chunk.replace("</think>", "")
            if status_callback:
                status_callback("Organizing concepts into map...")
        
        if in_think and status_callback:
            # Build up the thought and take the last ~60 chars
            current_thought += chunk
            # Clean up newlines for a cleaner single-line UI display
            clean_thought = current_thought.replace("\n", " ")
            if len(clean_thought) > 80:
                display_text = "..." + clean_thought[-77:]
            else:
                display_text = clean_thought
            status_callback(display_text.strip())
            
    if not response_text:
        print("Concept extraction: Ollama returned no response.", file=sys.stderr)
        return empty
    
    data = _parse_json_response(response_text)
    if not data:
        print(f"Concept extraction: Could not parse JSON from response.", file=sys.stderr)
        return empty
    
    graph = _validate_graph(data)
    print(f"Extracted {len(graph['nodes'])} nodes, {len(graph['edges'])} edges.", file=sys.stderr)
    return graph


def merge_concept_maps(
    session_maps: list[dict[str, Any]],
    model: str = "deepseek-r1:8b",
    status_callback: Callable[[str], None] | None = None,
    llm_provider: str = "ollama",
    api_key: str = ""
) -> dict[str, Any]:
    """Merge concept maps from multiple sessions using Ollama or Gemini.
    
    If status_callback is provided, it will be called with live <think> text updates.
    Args:
        session_maps: list of {"session_id", "title", "graph"} dicts
        
    Returns merged {"nodes": [...], "edges": [...]}
    """
    empty = {"nodes": [], "edges": []}
    
    graphs_with_content = [m for m in session_maps if m.get("graph", {}).get("nodes")]
    if not graphs_with_content:
        return empty
    
    if len(graphs_with_content) == 1:
        return graphs_with_content[0]["graph"]
    
    # Annotate nodes with their source session
    for m in graphs_with_content:
        for node in m["graph"].get("nodes", []):
            node["source_session"] = m.get("title", m.get("session_id", "unknown"))
    
    maps_json = json.dumps(
        [{"title": m.get("title", ""), "graph": m["graph"]} for m in graphs_with_content],
        indent=2,
    )
    
    # Truncate if too long
    if len(maps_json) > 15_000:
        maps_json = maps_json[:15_000] + "\n..."
    
    prompt = MERGE_PROMPT.replace("{maps_json}", maps_json)
    
    print(f"Merging concept maps via {llm_provider.capitalize()}...", file=sys.stderr)
    
    response_text = ""
    in_think = False
    current_thought = ""
    
    if llm_provider == "gemini":
        from main import ask_gemini_stream
        stream = ask_gemini_stream(prompt, api_key=api_key)
    else:
        stream = ask_ollama_stream(prompt, model=model)
        
    for chunk in stream:
        response_text += chunk
        
        # Determine if we're inside a <think> block
        if "<think>" in chunk:
            in_think = True
            chunk = chunk.replace("<think>", "")
        if "</think>" in chunk:
            in_think = False
            chunk = chunk.replace("</think>", "")
            if status_callback:
                status_callback("Organizing merged concepts...")
        
        if in_think and status_callback:
            current_thought += chunk
            clean_thought = current_thought.replace("\n", " ")
            if len(clean_thought) > 80:
                display_text = "..." + clean_thought[-77:]
            else:
                display_text = clean_thought
            status_callback(display_text.strip())
            
    if not response_text:
        # Fallback: naive merge without LLM
        return _naive_merge(graphs_with_content)
    
    data = _parse_json_response(response_text)
    if not data:
        return _naive_merge(graphs_with_content)
    
    graph = _validate_graph(data)
    print(f"Merged map: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges.", file=sys.stderr)
    return graph


def _naive_merge(session_maps: list[dict[str, Any]]) -> dict[str, Any]:
    """Fallback merge: combine all graphs, prefix IDs to avoid collisions."""
    all_nodes = []
    all_edges = []
    
    for i, m in enumerate(session_maps):
        prefix = f"s{i}_"
        graph = m.get("graph", {})
        for node in graph.get("nodes", []):
            new_node = dict(node)
            new_node["id"] = prefix + new_node["id"]
            new_node["source_session"] = m.get("title", f"Session {i+1}")
            all_nodes.append(new_node)
        for edge in graph.get("edges", []):
            new_edge = dict(edge)
            new_edge["source"] = prefix + new_edge["source"]
            new_edge["target"] = prefix + new_edge["target"]
            all_edges.append(new_edge)
    
    return {"nodes": all_nodes, "edges": all_edges}

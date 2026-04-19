import time
from datetime import datetime
from typing import List, Dict, Any

import regex as re
from loguru import logger


def render_results_html(
        operation: str,
        run_name: str,
        backend_name: str,
        results: List[Dict[str, Any]],
        output_path: str
):
    """Render results as a static HTML report."""
    logger.info(f'Rendering HTML report: title={operation}, count={len(results)}, path={output_path}')
    t0 = time.perf_counter()
    html = f"""
    <html>
    <head>
        <title>{operation} - {run_name}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            .header {{ background: #f4f4f4; padding: 10px; border-radius: 5px; }}
            .result-item {{ border-bottom: 1px solid #ddd; padding: 10px 0; }}
            .score {{ color: #666; font-size: 0.9em; }}
            .meta {{ color: #888; font-size: 0.8em; }}
            .text {{ margin-top: 5px; }}
            .seed-section {{ margin-top: 30px; }}
            h2 {{ border-left: 5px solid #007bff; padding-left: 10px; }}
            .highlight {{ background-color: #ffeb3b; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{operation}</h1>
            <p><strong>Run:</strong> {run_name}</p>
            <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
            <p><strong>Backend/Model:</strong> {backend_name}</p>
        </div>
    """

    # Group by seed or seed_regex
    grouped = {}
    for res in results:
        seed = res.get('seed') or res.get('seed_regex') or res.get('seed_term') or 'Results'
        if seed not in grouped:
            grouped[seed] = []
        grouped[seed].append(res)

    for seed, items in grouped.items():
        html += f'<div class="seed-section"><h2>Seed: {seed}</h2>'

        # Try to compile seed as regex for highlighting if it looks like one
        highlight_pattern = None
        if seed.startswith('/') and seed.endswith('/'):
            try:
                highlight_pattern = re.compile(seed[1:-1], re.IGNORECASE)
            except:
                pass
        elif any(c in seed for c in '*+?|()[]{}'):
            try:
                highlight_pattern = re.compile(seed, re.IGNORECASE)
            except:
                pass
        else:
            # literal escape for highlight
            highlight_pattern = re.compile(re.escape(seed), re.IGNORECASE)

        for item in sorted(items, key=lambda x: x.get('score', 0), reverse=True):
            score = item.get('score', 0.0)
            text = item.get('candidate') or item.get('expanded_term') or item.get('text') or ''

            # highlight seed in text if possible
            display_text = text
            if highlight_pattern:
                display_text = highlight_pattern.sub(lambda m: f'<span class="highlight">{m.group()}</span>', text)

            doc_id = item.get('doc_id', 'N/A')
            unit_id = (f'sent_id: {item.get("sent_id")}' if 'sent_id' in item
                       else f'para_id: {item.get("para_id")}' if 'para_id' in item else '')

            html += f"""
            <div class="result-item">
                <div class="score">Score: {score:.4f}</div>
                <div class="text">{display_text}</div>
                <div class="meta">doc_id: {doc_id} {unit_id}</div>
            """

            if 'relevant_sentences' in item:
                html += ('<div class="relevant-sentences" style="margin-left: 20px; margin-top: 10px; '
                         'font-size: 0.9em; border-left: 2px solid #eee; padding-left: 10px;">')
                html += '<strong>Relevant Sentences:</strong>'
                for sent in item['relevant_sentences']:
                    html += (f'<div style="margin-top: 5px;">- {sent["text"]} '
                             f'<span class="score" style="font-size: 0.8em;">({sent["score"]:.4f})</span>'
                             f'</div>')
                html += '</div>'

            html += "</div>"
        html += '</div>'

    html += '</body></html>'

    with open(output_path, 'w', encoding='utf8') as f:
        f.write(html)

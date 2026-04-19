from pattern_nursery.reports import html


def test_render_results_html(tmp_path):
    output = tmp_path / 'report.html'
    results = [
        {
            'seed_term': 'Väinämöinen',
            'expanded_term': 'sage',
            'score': 0.95,
            'doc_id': 'v_001'
        },
        {
            'seed_term': 'Väinämöinen',
            'expanded_term': 'hero',
            'score': 0.85,
            'doc_id': 'v_002'
        }
    ]

    html.render_results_html(
        operation='Expansion',
        run_name='test_run',
        backend_name='fake-bert',
        results=results,
        output_path=str(output)
    )

    assert output.exists()
    content = output.read_text(encoding='utf8')
    assert '<h1>Expansion</h1>' in content
    assert 'test_run' in content
    assert 'Seed: Väinämöinen' in content
    assert 'sage' in content
    assert 'hero' in content
    assert 'Score: 0.9500' in content


def test_render_empty_results(tmp_path):
    output = tmp_path / 'empty.html'
    html.render_results_html('Empty', 'none', 'none', [], str(output))
    assert output.exists()
    content = output.read_text(encoding='utf8')
    assert '<h1>Empty</h1>' in content

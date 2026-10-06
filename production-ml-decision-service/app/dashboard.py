DASHBOARD_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>ML Decision Service</title>
<style>body{font:15px system-ui;margin:2rem;max-width:1100px;color:#172033}h1{margin-bottom:.2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem}.card{border:1px solid #d7dce5;border-radius:10px;padding:1rem}.value{font-size:1.7rem;font-weight:700}table{width:100%;border-collapse:collapse;margin-top:1rem}th,td{text-align:left;padding:.6rem;border-bottom:1px solid #e3e6ec}.ok{color:#08783e}.muted{color:#667085}</style></head>
<body><h1>Production ML Decision Service</h1><p class="muted">Champion status, business threshold, and operational health</p>
<div id="cards" class="grid"></div><h2>Model details</h2><table id="details"></table>
<script>async function load(){const [m,h]=await Promise.all([fetch('/v1/model').then(r=>r.json()),fetch('/health/ready').then(r=>r.json())]);
const cards=[['Status',h.status],['Model',m.model_type],['Threshold',m.threshold.toFixed(2)],['PR-AUC',m.metrics.pr_auc.toFixed(3)],['Precision',m.metrics.precision.toFixed(3)],['Recall',m.metrics.recall.toFixed(3)]];
document.querySelector('#cards').innerHTML=cards.map(x=>`<div class="card"><div class="muted">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('');
document.querySelector('#details').innerHTML=Object.entries({version:m.model_version,dataset:m.dataset_version,capacity:m.capacity,net_value:m.metrics.expected_net_value}).map(x=>`<tr><th>${x[0]}</th><td>${x[1]}</td></tr>`).join('');}load();</script></body></html>
"""


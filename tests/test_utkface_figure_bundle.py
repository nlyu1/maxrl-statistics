import argparse
import json
import re
import runpy
from pathlib import Path
from typing import TYPE_CHECKING

from plotly.graph_objects import Figure

if TYPE_CHECKING:
    import quickjs

HARNESS = r"""
document.getElementById('bundle-data').textContent=snapshot_json;
document.getElementById('navigation').replaceChildren();
for(const [id,value] of Object.entries({'smoothing':'auto','seed':'all','x-mode':'epoch'}))document.getElementById(id).value=value;
document.getElementById('bands').checked=true;
for(const value of ['all','42','43','44']){const o=new Element('option');o.value=value;document.getElementById('seed').append(o);}
window.location={hash:''};window.addEventListener=()=>{};
function setDeep(obj,key,value){const parts=key.split('.');let item=obj;for(const p of parts.slice(0,-1))item=item[p]||(item[p]={});item[parts[parts.length-1]]=value;}
const Plotly={
 react(div,data,layout,config){div.data=data;div.layout=layout;div.config=config;div.events=div.events||{};div.on=(event,fn)=>{div.events[event]=fn;};return Promise.resolve();},
 relayout(div,changes){for(const [k,v] of Object.entries(changes))setDeep(div.layout,k,v);if(div.events.plotly_relayout)div.events.plotly_relayout(changes);return Promise.resolve();},
 restyle(div,changes,indices){for(const i of indices)Object.assign(div.data[i],changes);if(div.events.plotly_restyle)div.events.plotly_restyle(changes);return Promise.resolve();}
};
function check(value,message){if(!value)throw Error(message);}
function change(id,value){document.getElementById(id).value=value;for(const f of document.getElementById(id).listeners.change||[])f({});}
function click(id){for(const f of document.getElementById(id).listeners.click||[])f({preventDefault(){}});}
"""

ASSERTIONS = r"""
check(B.figures.length===7,'seven pages');
check(document.getElementById('navigation').children.length===7,'seven navigation links');
check(new Set(graph.data.filter(t=>t.meta.kind==='mean').map(t=>t.meta.key)).size===8,'eight main methods');
check(graph.data.filter(t=>t.showlegend).length===8,'one main legend item per method');
check(graph.layout.legend.groupclick==='togglegroup'&&graph.layout.legend.itemdoubleclick==='toggleothers','legend group toggling and isolation');
const scalarNll=graph.data.filter(t=>t.meta.panel===3&&t.meta.key==='scalar_mae');check(!scalarNll.length,'undefined scalar NLL omitted');
const r=B.runs[uniqueGroups(current()).find(g=>g.key==='scalar_mae').ids[0]];
const firstExpected=r.curves.slice(0,5).reduce((s,c)=>s+c[1],0)/5;
const firstCurve=curve({ids:[r.id]},'mae_mean',50);check(Math.abs(firstCurve.mean[0]-firstExpected)<1e-12&&firstCurve.x[0]===3,'moving mean and centered epochs');
change('smoothing','1');check(graph.data.find(t=>t.meta.kind==='mean').x.length===50,'raw epochs');
change('smoothing','auto');
Plotly.relayout(graph,{'xaxis.range':[10,40],'yaxis.range':[4,7]});
check(pageState().ranges.xaxis[0]===10&&pageState().ranges.yaxis[1]===7,'native Plotly zoom saved');
change('smoothing','10');check(graph.layout.xaxis.range[0]===10&&graph.layout.yaxis.range[1]===7,'native zoom survives smoothing');
change('smoothing','auto');
for(const trace of graph.data)if(trace.meta.key==='tailrl')trace.visible='legendonly';captureLegend();check(pageState().hidden.has('tailrl'),'legend selection retained');
render();check(graph.data.filter(t=>t.meta.key==='tailrl'&&t.meta.kind==='mean').every(t=>t.visible==='legendonly'),'hidden methods survive rerender');
state.bands=false;document.getElementById('bands').checked=false;render();check(graph.data.filter(t=>t.meta.kind==='band').every(t=>t.visible===false),'SD visibility');
for(const f of B.figures){setPage(f.id);check(!document.getElementById('render-error').textContent,'page render error');check(graph.data.filter(t=>t.meta.kind==='mean').every(t=>t.x.length===t.y.length&&t.y.every(Number.isFinite)),'finite matched traces');}
setPage('03-main-1024');check(new Set(graph.data.filter(t=>t.meta.kind==='mean').map(t=>t.meta.key)).size===8,'eight actual 1024 methods');
check(graph.data.filter(t=>t.meta.kind==='mean').every(t=>t.meta.n===1),'1024 seed count');
check(document.getElementById('seed').children.find(n=>n.value==='43').disabled,'unavailable seeds disabled');
change('smoothing','1');check(graph.data.find(t=>t.meta.kind==='mean').x.length===200,'all 200 raw epochs');
change('x-mode','updates');const updatePoints=graph.data.find(t=>t.meta.kind==='mean').x;check(updatePoints[updatePoints.length-1]===3200,'optimizer update axis');
change('x-mode','epoch');change('smoothing','auto');
setPage('06-fixed-noise');check(current().panels.length===16,'sixteen noise panels');
check(new Set(current().panels.flatMap(p=>p.groups.map(g=>g.key))).size===5,'five noise methods');
check(current().panels[4].title.startsWith('SD 3'),'approved SD3 substitution');
setPage('05-tail-focus');check(current().panels.length===4,'full-data focus only');
check(current().panels.every(p=>p.groups.every(g=>g.ids.every(id=>B.runs[id].size===18462&&B.runs[id].epochs===50))),'focus uses full-data runs');
setPage('04-stat-widths');check(current().panels.length===8&&uniqueGroups(current()).length===9,'widths and estimators');
check(graph.data.filter(t=>t.showlegend).length===5,'five shared width legend entries');
check(new Set(graph.data.filter(t=>t.meta.kind==='mean').map(t=>t.legendgroup)).size===5,'five shared legend groups');
check(graph.data.filter(t=>t.meta.kind==='mean'&&t.meta.panel===0).length===5&&graph.data.filter(t=>t.meta.kind==='mean'&&t.meta.panel===4).length===5,'five curves per estimator panel');
const widthKey=graph.data.find(t=>t.showlegend&&t.meta.methodKey.startsWith('stat_')).meta.key;
for(const trace of graph.data)if(trace.legendgroup===widthKey)trace.visible='legendonly';captureLegend();render();
check(graph.data.filter(t=>t.meta.key===widthKey&&t.meta.kind==='mean').length===8&&graph.data.filter(t=>t.meta.key===widthKey&&t.meta.kind==='mean').every(t=>t.visible==='legendonly'),'shared legend hides kernel and width across both rows');
check(new Set(graph.data.filter(t=>t.meta.key===widthKey&&t.meta.kind==='mean').map(t=>t.meta.methodKey)).size===2,'shared legends retain separate estimator provenance');
setPage('07-digits');check(uniqueGroups(current()).length===8&&current().panels.every(p=>p.groups.every(g=>g.ids.every(id=>B.runs[id].head==='digits'))),'eight actual digit objectives');
check(current().panels.length===4&&current().panels.every(p=>p.groups.every(g=>g.ids.every(id=>B.runs[id].size===18462&&B.runs[id].epochs===50))),'digit figure uses full data only');
click('fit-all');check(Object.values(graph.layout).filter(v=>v&&v.range).every(v=>v.range[0]<v.range[1]),'full trajectory fitting');
click('reset');check(pageState().hidden.size===0&&Object.keys(pageState().ranges).length===0,'reset view');
"""


def drain(*, context: "quickjs.Context") -> None:
    while context.execute_pending_job():
        pass


def verify(*, html_path: Path) -> dict:
    import quickjs

    html = html_path.read_text()
    data = re.search(
        r'<script id="bundle-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    ).group(1)
    script = re.search(
        r'<script id="bundle-app">(.*?)</script>', html, re.DOTALL
    ).group(1)
    shared = runpy.run_path(
        str(Path(__file__).with_name("utkface_report_interactions.py"))
    )["DOM_HARNESS"]
    context = quickjs.Context()
    context.set("snapshot_json", data)
    context.eval(shared)
    context.eval(HARNESS)
    context.eval(script)
    drain(context=context)
    for statement in ASSERTIONS.splitlines():
        if statement.strip():
            context.eval(statement)
            drain(context=context)
    payload = json.loads(data)
    for fig in payload["figures"]:
        context.eval(f"setPage({json.dumps(fig['id'])});")
        drain(context=context)
        plot = json.loads(
            context.eval("JSON.stringify({data:graph.data,layout:graph.layout})")
        )
        Figure(data=plot["data"], layout=plot["layout"], skip_invalid=False)
        assert len(fig["panels"]) == fig["rows"] * fig["cols"]
        for panel in fig["panels"]:
            for group in panel["groups"]:
                for identifier in group["ids"]:
                    assert identifier in payload["runs"]
    assert "__PLOTLY_RUNTIME__" not in html and "__BUNDLE_DATA__" not in html
    assert not re.search(r"<script[^>]+src=", html)
    assert 'id="axis-target"' not in html and 'id="apply-ranges"' not in html
    return {
        "pages": len(payload["figures"]),
        "sourceRuns": len(payload["runs"]),
        "sourceSha256": payload["sourceSha256"],
        "result": "passed",
        "checks": [
            "seven pages and navigation",
            "method coverage and legend groups",
            "scalar NLL omitted",
            "raw and centered moving means",
            "seed coverage",
            "native Plotly zoom persistence",
            "custom axis toolbar removed",
            "legend state retention",
            "SD bands",
            "finite traces",
            "200 epochs and optimizer updates",
            "16 noise panels and SD3 adaptation",
            "full-data focus provenance",
            "five shared width legends and cross-row toggling",
            "distinct estimator provenance",
            "full-data digit policy provenance",
            "full-range fit and reset",
            "offline runtime",
            "all seven figures pass the installed Plotly schema",
        ],
        "browserRenderingVerified": False,
        "scope": "Isolated DOM and Plotly API interaction contract; static PNGs inspected separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(html_path=args.html)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

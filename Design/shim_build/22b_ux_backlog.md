you’re right — the demo feels a bit “wonky.” let’s tighten it so every pixel earns its keep. below is a **surgical redesign** of your HUD that:

* ditches gimmicks (particles, random colors),
* speaks plain language,
* shows the **math you built** and the **stability story** side-by-side,
* and wires directly to the signals your runtime already emits (no new theory).

i’m keeping it short, composable, and anime.js-friendly so you can drop it in and iterate.



---

# what we’ll change (quick and concrete)

1. **Language → plain words**

* “Coupling index” (maps to η_dd), “Settling rate” (maps to γ), “Score” (energy E), “View” (lens), “Detail level” (active scale), “Health” (RED/AMBER/GREEN).
* Remove jargon on screen; keep math in a neat LaTeX line.

2. **Layout → two clear cards**

* **Math Card (left):** your score formula + term chips with live weights and who’s “pulling” (grad share).
* **Stability Card (right):** Health light, two small gauges (Coupling index / Settling rate), score sparkline, complexity row (Flow size, Detail level, View), and a human-readable event ticker.

3. **Animations → meaning only**

* Health flips on phase change; gauges shake/flicker when out of bounds; term chips pulse ∝ gradient share; lens selection “focus” zoom; scale rings light up on activation.
* No confetti/particles; color encodes state and nothing else.

---

# trimmed HUD scaffold (drop-in)

> Replace the current HTML with this skeleton. It keeps your anime.js setup but binds to a **single** `updateTelemetry(data)` function so you can stream real rows later.

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Flow Card</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://unpkg.com/animejs@3.2.1/lib/anime.min.js"></script>
  <style>
    :root{--bg:#0b0d12;--fg:#e6e9ef;--mut:#9aa3af;--panel:#111827;--line:#243041;--good:#10b981;--warn:#f59e0b;--bad:#ef4444;--accent:#60a5fa}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
    .wrap{display:grid;grid-template-columns:1fr 1fr;gap:24px;max-width:1200px;margin:24px auto;padding:0 16px}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
    h3{margin:0 0 8px 0;font-weight:700}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    .pill{border:1px solid var(--line);background:#0e1522;border-radius:999px;padding:6px 10px;font-size:12px}
    .math{font-family: ui-monospace,SFMono-Regular,Menlo,monospace;background:#0f172a;border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow:auto}
    .light{width:14px;height:14px;border-radius:50%}
    .g{background:var(--good)} .y{background:var(--warn)} .r{background:var(--bad)}
    .gauge{width:120px;height:120px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--accent) 0deg,#1f2937 0deg);border:1px solid var(--line)}
    .gauge span{font-size:12px;color:var(--mut);margin-top:6px;display:block;text-align:center}
    .kv{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;font-size:13px;margin-top:8px;color:var(--mut)}
    .spark{height:48px;border:1px solid var(--line);border-radius:6px;margin-top:8px;position:relative;overflow:hidden}
    .spark path{stroke:var(--good);stroke-width:2;fill:none}
    .ticker{margin-top:12px;border:1px solid var(--line);border-radius:6px;max-height:120px;overflow:auto;padding:8px;font-size:12px;color:var(--mut)}
    .chips{display:flex;gap:8px;flex-wrap:wrap}
    .chip{position:relative} .chip-pulse{position:absolute;left:0;top:0;bottom:0;width:0;background:#60a5fa20;border-radius:999px}
    .sub{font-size:12px;color:var(--mut)}
    .cols{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px}
    .col{display:flex;flex-direction:column;align-items:center}
  </style>
</head>
<body>
  <div class="wrap">
    <!-- Math Card -->
    <div class="card" id="mathCard">
      <h3>Score formula</h3>
      <div class="math" id="mathLatex">E(x) = ½‖Ax−y‖² + λ‖Wx‖₁ + τ·TV(x)</div>
      <div class="sub" style="margin-top:6px">Balancing <b>fit to data</b>, <b>sparsity</b>, and <b>smoothness</b>.</div>
      <div class="chips" id="termChips" style="margin-top:10px"></div>
      <div class="sub" style="margin-top:10px">Flow: <span id="flowChain">descend → tidy → change view</span></div>
    </div>

    <!-- Stability Card -->
    <div class="card">
      <h3>Stability</h3>
      <div class="row">
        <div class="light g" id="healthLight"></div>
        <div id="healthText">GREEN · settling</div>
      </div>

      <div class="cols" style="margin-top:10px">
        <div class="col">
          <div class="gauge" id="couplingGauge"></div>
          <span>Coupling index (lower is safer)</span>
        </div>
        <div class="col">
          <div class="gauge" id="settlingGauge"></div>
          <span>Settling rate (higher is better)</span>
        </div>
      </div>

      <div class="kv">
        <div>Flow size</div><div id="flowSize">3 terms</div>
        <div>Detail level</div><div id="detailLevel">Level 3 / 6</div>
        <div>View</div><div id="lensName">db4</div>
      </div>

      <div class="sub" style="margin-top:8px">Score trend</div>
      <div class="spark"><svg id="sparkSvg" width="100%" height="100%"><path id="sparkPath" d="M0,40 L20,35 L40,30 L60,24 L80,21 L100,20"/></svg></div>

      <div class="sub" style="margin-top:10px">Events</div>
      <div class="ticker" id="ticker"><div>Run started</div></div>
    </div>
  </div>

  <script>
    const H = {
      setHealth(phase){
        const l = document.getElementById('healthLight');
        const t = document.getElementById('healthText');
        l.className = 'light ' + (phase==='GREEN'?'g':phase==='AMBER'?'y':'r');
        const text = phase==='GREEN'?'GREEN · settling':phase==='AMBER'?'AMBER · unsure':'RED · rollback';
        anime({ targets: t, rotateY:[0,90,0], duration:360, easing:'easeInOutQuad', update:a=>{ if(a.progress>50) t.textContent = text; }});
      },
      setGauge(el, value, targetGood='low'){
        // value normalized to [0,1]. Conic fill + attention cues
        const deg = Math.max(0, Math.min(360, value*360));
        el.style.background = `conic-gradient(var(--accent) ${deg}deg, #1f2937 ${deg}deg)`;
        if ((targetGood==='low' && value>0.9) || (targetGood==='high' && value<0.1)){
          anime({targets: el, translateX:[0,-6,6,-4,4,0], duration:420, easing:'easeInOutQuad'});
        }
      },
      setSpark(values){
        const svg = document.getElementById('sparkSvg');
        const path = document.getElementById('sparkPath');
        const w = svg.clientWidth, h = svg.clientHeight;
        const ys = values.map(v => h - v*h*0.8);
        const xs = values.map((_,i)=> (i/(values.length-1||1))*w);
        const d = 'M'+xs.map((x,i)=>`${x},${ys[i]}`).join(' L ');
        anime({targets:path, d, duration:360, easing:'easeInOutQuad'});
      },
      setChips(terms){
        const wrap = document.getElementById('termChips'); wrap.innerHTML='';
        terms.forEach(term=>{
          const c = document.createElement('div'); c.className='pill chip';
          c.textContent = `${term.name} · ${term.weight}`;
          const pulse = document.createElement('div'); pulse.className='chip-pulse'; c.appendChild(pulse);
          wrap.appendChild(c);
          if (term.grad_share && term.grad_share>0){
            anime.set(pulse,{width:0, opacity:0});
            anime({targets:pulse, width:(100*term.grad_share)+'%', opacity:[0,.5,0], duration:600, easing:'easeInOutSine'});
          }
        });
      },
      tick(evt){ // simple event sink
        const tk = document.getElementById('ticker'); const d = document.createElement('div'); d.textContent = evt; tk.prepend(d);
        anime({targets:d, opacity:[0,1], translateX:[-16,0], duration:320, easing:'easeOutQuad'});
      }
    };

    // --- bind to telemetry row ---
    function updateTelemetry(row){
      // Health
      H.setHealth(row.phase||'AMBER');

      // Gauges
      const eta = row.eta_dd ?? 0.7;           // 0..1 where lower=better
      const gam = row.gamma ?? 1e-6;           // map to 0..1 versus threshold
      const gamNorm = Math.min(1, (gam / ((row.gamma_min||1e-6)*2 || 1e-9)));
      H.setGauge(document.getElementById('couplingGauge'), eta, 'low');
      H.setGauge(document.getElementById('settlingGauge'), 1-gamNorm, 'low'); // invert so low=bad

      // Complexity
      document.getElementById('flowSize').textContent = (row.active_terms?.length||0)+' terms';
      document.getElementById('detailLevel').textContent = 'Level '+(row.level_active_max??0);
      const lens = row.lens_name||'—';
      const lensEl = document.getElementById('lensName');
      if (lensEl.textContent !== lens){
        anime({targets:lensEl, scale:[1,1.15,1], duration:260, easing:'easeOutQuad'});
        lensEl.textContent = lens;
      }

      // Score sparkline
      window._Ehist = (window._Ehist||[]).concat([row.E_norm ?? 0.5]).slice(-50);
      H.setSpark(window._Ehist);

      // Terms
      H.setChips(row.active_terms||[]);

      // Events (optional)
      if (row.event_text) H.tick(row.event_text);
    }

    // --- demo ping (remove when wiring real stream) ---
    let t=0; setInterval(()=> {
      t++;
      updateTelemetry({
        phase: t<6?'AMBER': (t%11===0?'RED':'GREEN'),
        eta_dd: 0.65 + 0.25*Math.abs(Math.sin(t/10)),
        gamma: 1e-6 + 6e-7*Math.sin(t/7) + 1.2e-6,
        lens_name: (t%15===0)?'db4':'db4',
        level_active_max: 2 + (t%4),
        E_norm: Math.max(0,.9- t*0.015 + 0.05*Math.sin(t/3)),
        active_terms: [
          {name:'Fit to data', weight: '1.0', grad_share: 0.6+0.2*Math.sin(t/5)},
          {name:'Sparsity', weight: (0.02).toFixed(2), grad_share: 0.25+0.15*Math.cos(t/6)},
          {name:'TV', weight: (0.01).toFixed(2), grad_share: 0.15+0.1*Math.sin(t/4)}
        ],
        event_text: (t%9===0)?'Tuned sparsity: change rejected (coupling too high)':''
      });
    }, 800);
  </script>
</body>
</html>
```

---

## how this fixes the “wonky” bits

* **Single source of truth:** everything updates through `updateTelemetry(row)`. When you hook your `telemetry.parquet` stream to this, the HUD just works.
* **Signals over spectacle:** gauges, pulses, and flips are tied to *certification & controller events*, not arbitrary sliders.
* **Plain words:** no Greek on screen; math is shown once, cleanly.
* **Composable:** the “Math Card” and “Stability Card” are independent — you can slot them into any page.

want me to wire this to your existing JSON field names and produce a tiny adapter that reads rows from DuckDB and pushes them to the page via a websocket?

"""Build docs/blog.html (Blogger-ready edition) from docs/book.html.

Transformations:
- extracts the book content (no navbar, sidebar, search, progress, scripts)
- converts Bootstrap accordions to native <details> elements (zero JS)
- replaces the Bootstrap hero stats grid with scoped markup
- generates an in-post table of contents with anchor links
- scopes every style rule under .fxbook so the post cannot clash with a blog theme

Run:  python tools/build_blog.py
"""

import io
import re

SRC = "docs/book.html"
DST = "docs/blog.html"

s = io.open(SRC, encoding="utf-8").read()

# ---- 1. extract the content between the content wrapper -------------------
start = s.index('<div class="fx-content" id="top">') + len('<div class="fx-content" id="top">')
end = s.index('</div><!-- /fx-content -->')
content = s[start:end]

# ---- 2. hero stats: replace the Bootstrap grid with scoped markup ---------
stats_re = re.compile(r'<div class="row row-cols-2 row-cols-md-4 g-3 mt-2">.*?</header>', re.S)
stats_new = (
    '<div class="fx-stats">\n'
    '      <div class="fx-stat"><div class="n">14</div><small>parts</small></div>\n'
    '      <div class="fx-stat"><div class="n">41</div><small>chapters</small></div>\n'
    '      <div class="fx-stat"><div class="n">0</div><small>assumed prerequisites</small></div>\n'
    '      <div class="fx-stat"><div class="n">16&nbsp;GB</div><small>enough to begin</small></div>\n'
    '    </div>\n'
    '  </header>'
)
content, n = stats_re.subn(stats_new, content)
assert n == 1, "hero stats not replaced"

# ---- 3. convert both Bootstrap accordions to native <details> -------------
item_re = re.compile(
    r'<div class="accordion-item">\s*'
    r'<h3 class="accordion-header"><button class="accordion-button(?: collapsed)?" type="button" '
    r'data-bs-toggle="collapse" data-bs-target="#([\w-]+)">(.*?)</button></h3>\s*'
    r'<div id="\1" class="accordion-collapse collapse(?: show)?" data-bs-parent="#[\w-]+">'
    r'<div class="accordion-body">'
    r'(.*?)'
    r'</div></div>\s*</div>', re.S)


def item_sub(m):
    fid, title, body = m.group(1), m.group(2), m.group(3)
    open_attr = ' open' if fid in ("m1", "f1") else ''  # first item of each group was open
    return (f'<details class="fx-acc"{open_attr}><summary>{title}</summary>'
            f'<div class="fx-acc-body">{body}</div></details>')


content, n_items = item_re.subn(item_sub, content)
assert n_items == 25, f"expected 25 accordion items, got {n_items}"
content = content.replace('<div class="accordion" id="miscAcc">', '<div class="fx-acc-group">')
content = content.replace('<div class="accordion" id="faqAcc">', '<div class="fx-acc-group">')

# ---- 4. remove the JS-dependent glossary filter ---------------------------
content, n = re.subn(r'<input class="form-control form-control-sm mb-3" id="glossary-filter"[^>]*/?>\s*', '', content)
assert n == 1, "glossary filter not removed"

# ---- 5. build an in-post table of contents --------------------------------
token_re = re.compile(
    r'<div class="part-divider" id="[^"]+">\s*<div class="part-label">([^<]+)</div>\s*<h2>([^<]+)</h2>'
    r'|<section class="chapter" id="([\w-]+)" data-chapter data-nav-title="([^"]+)">')

toc_groups, chapter_num = [], 0
for m in token_re.finditer(content):
    if m.group(1) is not None:
        toc_groups.append({"label": m.group(1), "title": m.group(2), "chapters": []})
    else:
        cid, ctitle = m.group(3), m.group(4)
        if not toc_groups:  # "How to Read This Book" appears before Part I
            toc_groups.append({"label": "Start here", "title": "", "chapters": []})
        num = ''
        if cid.startswith("ch-") and cid[3:].isdigit():
            chapter_num = int(cid[3:])
            num = f'{chapter_num}. '
        toc_groups[-1]["chapters"].append((cid, num + ctitle))

toc_html = ['<nav class="fx-toc-box"><div class="fx-toc-title">Contents</div>']
for g in toc_groups:
    toc_html.append(f'<p class="toc-part">{g["label"]} &#8212; {g["title"]}</p><p class="toc-chs">' +
                    ''.join(f'<a href="#{cid}">{t}</a>' for cid, t in g["chapters"]) + '</p>')
toc_html.append('</nav>')
toc_html = '\n'.join(toc_html)

hero_end = content.index('</header>') + len('</header>')
content = content[:hero_end] + '\n\n' + toc_html + '\n' + content[hero_end:]

# strip attributes that only the book's JavaScript used
content = content.replace(' data-chapter', '')
content = re.sub(r' data-nav-title="[^"]*"', '', content)

# ---- 6. scoped stylesheet --------------------------------------------------
css = '''
    /* Fontaine AI, blog edition. Every rule is scoped under .fxbook
       so the post can never clash with the host blog theme. */
    .fxbook{
      --fx-accent:#0e7490; --fx-accent-soft:#e0f2f7; --fx-amber:#b45309;
      --fx-paper:#fbfaf8; --fx-surface:#ffffff; --fx-ink:#1f2937; --fx-muted:#5b6776;
      --fx-line:#e7e2d8; --fx-code-bg:#f4f4f2; --fx-shadow:0 10px 30px rgba(31,41,55,.07);
      background:var(--fx-paper); color:var(--fx-ink);
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue","Noto Sans",Arial,sans-serif;
      font-size:1.02rem; line-height:1.75;
      max-width:900px; margin:0 auto; padding:1.4rem 1.1rem 3rem;
      border-radius:16px; box-sizing:border-box;
    }
    .fxbook,.fxbook *{box-sizing:border-box;}
    .fxbook h2,.fxbook h3,.fxbook h4{font-family:Georgia,"Times New Roman","Noto Serif",serif;letter-spacing:-.01em;}
    .fxbook h3{margin-top:2rem;font-size:1.28rem;font-weight:700;}
    .fxbook h4{margin-top:1.4rem;font-size:1.08rem;font-weight:700;}
    .fxbook p{margin-bottom:1.05rem;}
    .fxbook a{color:var(--fx-accent);text-decoration:none;}
    .fxbook a:hover{text-decoration:underline;}
    .fxbook ::selection{background:var(--fx-accent);color:#fff;}
    .fxbook small{font-size:.85em;}
    .fxbook .small{font-size:.85em;}
    .fxbook code{font-family:ui-monospace,"Cascadia Code",Consolas,"Courier New",monospace;}
    .fxbook p code,.fxbook li code,.fxbook td code{background:var(--fx-code-bg);border:1px solid var(--fx-line);border-radius:6px;padding:.08rem .38rem;font-size:.84em;}
    .fxbook pre{background:var(--fx-code-bg);border:1px solid var(--fx-line);border-radius:12px;padding:1rem 1.15rem;font-size:.84rem;line-height:1.6;overflow-x:auto;}
    .fxbook .term{border-bottom:2px dotted rgba(14,116,144,.6);cursor:help;font-weight:600;}

    .fxbook .fx-hero{background:radial-gradient(900px 400px at 85% -10%,rgba(14,116,144,.14),transparent),var(--fx-surface);border:1px solid var(--fx-line);border-radius:22px;padding:clamp(1.5rem,4vw,3rem);box-shadow:var(--fx-shadow);margin-bottom:2rem;}
    .fxbook .fx-hero .kicker{text-transform:uppercase;letter-spacing:.18em;font-size:.78rem;font-weight:700;color:var(--fx-accent);}
    .fxbook .fx-hero h1{font-size:clamp(1.9rem,5vw,3rem);line-height:1.12;margin:.6rem 0 1rem;}
    .fxbook .fx-hero .lead{font-size:1.1rem;max-width:46rem;color:var(--fx-muted);}
    .fxbook .fx-stats{display:flex;flex-wrap:wrap;gap:1.2rem;margin-top:1.2rem;}
    .fxbook .fx-stat{border-left:3px solid var(--fx-accent);padding-left:.8rem;}
    .fxbook .fx-stat .n{font-size:1.35rem;font-weight:700;font-family:Georgia,serif;}
    .fxbook .fx-stat small{color:var(--fx-muted);}

    .fxbook .fx-toc-box{background:var(--fx-surface);border:1px solid var(--fx-line);border-radius:14px;padding:1.1rem 1.3rem;margin:0 0 1rem;}
    .fxbook .fx-toc-title{font-family:Georgia,serif;font-weight:700;font-size:1.15rem;margin-bottom:.4rem;}
    .fxbook .toc-part{margin:.7rem 0 .15rem;font-weight:800;font-size:.82rem;text-transform:uppercase;letter-spacing:.1em;color:var(--fx-accent);}
    .fxbook .toc-chs{margin:0 0 .4rem;line-height:2;}
    .fxbook .toc-chs a{display:inline-block;background:var(--fx-accent-soft);border-radius:8px;padding:.12rem .55rem;margin:0 .25rem .25rem 0;font-size:.86rem;color:var(--fx-ink);}

    .fxbook section.chapter{padding-top:2.4rem;scroll-margin-top:10px;}
    .fxbook .chapter-kicker{text-transform:uppercase;letter-spacing:.16em;font-size:.74rem;font-weight:700;color:var(--fx-accent);}
    .fxbook .chapter-title{font-size:clamp(1.5rem,3vw,2.05rem);margin:.3rem 0 .4rem;}
    .fxbook .chapter-intro{font-size:1.05rem;color:var(--fx-muted);}
    .fxbook .chapter-id-chip{display:inline-block;background:var(--fx-accent);color:#fff;font-family:inherit;font-size:.8rem;font-weight:700;border-radius:999px;padding:.18rem .75rem;margin-right:.55rem;vertical-align:middle;}
    .fxbook .part-divider{margin:3.5rem 0 .5rem;padding:1.5rem 1.7rem;border-radius:14px;background:linear-gradient(120deg,rgba(14,116,144,.12),var(--fx-surface));border:1px solid var(--fx-line);scroll-margin-top:10px;}
    .fxbook .part-divider .part-label{text-transform:uppercase;letter-spacing:.2em;font-size:.75rem;font-weight:700;color:var(--fx-accent);}
    .fxbook .part-divider h2{margin:.2rem 0 .4rem;font-size:clamp(1.45rem,3vw,1.95rem);}
    .fxbook .part-divider p{margin:0;color:var(--fx-muted);max-width:52rem;}

    .fxbook .wwhw{display:grid;grid-template-columns:1fr;gap:.8rem;margin:1.2rem 0;}
    @media (min-width:720px){.fxbook .wwhw{grid-template-columns:1fr 1fr;}}
    .fxbook .wwhw .card{border:1px solid var(--fx-line);background:var(--fx-surface);border-radius:14px;}
    .fxbook .wwhw .card .card-body{padding:.95rem 1.1rem;}
    .fxbook .wwhw .wwhw-label{display:inline-flex;align-items:center;gap:.45rem;font-size:.74rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--fx-accent);}
    .fxbook .wwhw .card.wide{grid-column:1/-1;}

    .fxbook .callout{border:1px solid var(--fx-line);border-left-width:4px;border-radius:12px;background:var(--fx-surface);padding:.95rem 1.15rem;margin:1.25rem 0;}
    .fxbook .callout .callout-title{display:flex;align-items:center;gap:.5rem;font-weight:800;font-size:.84rem;text-transform:uppercase;letter-spacing:.09em;margin-bottom:.45rem;}
    .fxbook .callout p:last-child{margin-bottom:0;}
    .fxbook .callout ul:last-child{margin-bottom:0;}
    .fxbook .co-key{border-left-color:var(--fx-accent);}.fxbook .co-key .callout-title{color:var(--fx-accent);}
    .fxbook .co-why{border-left-color:#7c3aed;}.fxbook .co-why .callout-title{color:#7c3aed;}
    .fxbook .co-beg{border-left-color:#059669;}.fxbook .co-beg .callout-title{color:#059669;}
    .fxbook .co-tech{border-left-color:#475569;}.fxbook .co-tech .callout-title{color:#475569;}
    .fxbook .co-ana{border-left-color:var(--fx-amber);}.fxbook .co-ana .callout-title{color:var(--fx-amber);}
    .fxbook .co-honest{border-left-color:#dc2626;background:rgba(220,38,38,.04);}
    .fxbook .co-honest .callout-title{color:#dc2626;}

    .fxbook .fx-diagram{background:var(--fx-surface);border:1px solid var(--fx-line);border-radius:14px;box-shadow:var(--fx-shadow);padding:1.4rem 1.2rem .6rem;margin:1.4rem 0;}
    .fxbook .fx-diagram figcaption{font-size:.84rem;color:var(--fx-muted);border-top:1px dashed var(--fx-line);margin-top:1rem;padding:.7rem .2rem .3rem;}
    .fxbook .fx-diagram figcaption b{color:var(--fx-ink);}
    .fxbook .flow{display:flex;flex-direction:column;align-items:center;}
    .fxbook .flow-row{display:flex;flex-wrap:wrap;justify-content:center;gap:.6rem;width:100%;}
    .fxbook .fnode{border:1.5px solid var(--fx-accent);color:var(--fx-ink);background:rgba(14,116,144,.07);border-radius:10px;padding:.45rem .9rem;text-align:center;font-size:.88rem;font-weight:600;line-height:1.35;max-width:260px;}
    .fxbook .fnode.small{font-size:.8rem;font-weight:500;}
    .fxbook .fnode.accent{background:var(--fx-accent);color:#fff;border-color:var(--fx-accent);}
    .fxbook .fnode.warm{border-color:var(--fx-amber);background:rgba(180,83,9,.1);}
    .fxbook .fnode.ghost{border-style:dashed;background:transparent;}
    .fxbook .farrow{color:var(--fx-accent);font-size:1.05rem;line-height:1;padding:.28rem 0;text-align:center;user-select:none;}
    .fxbook .farrow.h{width:34px;align-self:center;}
    .fxbook .fnote{font-size:.8rem;color:var(--fx-muted);text-align:center;max-width:430px;margin:0 auto;}

    .fxbook .fx-table{border:1px solid var(--fx-line);border-radius:12px;overflow-x:auto;margin:1.2rem 0;}
    .fxbook .fx-table table{margin:0;border-collapse:collapse;width:100%;}
    .fxbook .fx-table thead th{background:rgba(14,116,144,.09);border-bottom:1px solid var(--fx-line);font-size:.86rem;text-align:left;}
    .fxbook .fx-table td,.fxbook .fx-table th{border-bottom:1px solid var(--fx-line);font-size:.93rem;padding:.6rem .85rem;vertical-align:middle;}
    .fxbook .fx-table tr:last-child td{border-bottom:none;}

    .fxbook .vs-grid{display:grid;grid-template-columns:1fr;gap:.8rem;margin:1.2rem 0;}
    @media (min-width:720px){.fxbook .vs-grid{grid-template-columns:1fr 1fr;}}
    .fxbook .vs-card{border:1px solid var(--fx-line);border-radius:14px;background:var(--fx-surface);padding:1rem 1.15rem;}
    .fxbook .vs-card h5{font-size:1rem;font-weight:800;margin:0 0 .5rem;display:flex;align-items:center;gap:.5rem;}
    .fxbook .vs-card.good{border-top:3px solid var(--fx-accent);}
    .fxbook .vs-card.bad{border-top:3px solid #dc2626;}
    .fxbook .vs-card ul{padding-left:1.1rem;margin-bottom:0;font-size:.93rem;}

    .fxbook .mathbox{background:var(--fx-surface);border:1px solid var(--fx-line);border-radius:14px;padding:1.1rem 1.3rem;margin:1.2rem 0;overflow-x:auto;}
    .fxbook .mathbox .formula{font-family:Georgia,serif;font-size:1.15rem;text-align:center;padding:.5rem 0 .2rem;letter-spacing:.02em;}
    .fxbook .mathbox .symbols{font-size:.88rem;margin-top:.6rem;border-collapse:collapse;}
    .fxbook .mathbox .symbols td{padding:.18rem .6rem .18rem 0;vertical-align:top;}

    .fxbook .fx-timeline{position:relative;margin:1.4rem .4rem 1.4rem;padding-left:1.6rem;border-left:2.5px solid rgba(14,116,144,.45);}
    .fxbook .fx-timeline .tl-item{position:relative;padding-bottom:1.3rem;}
    .fxbook .fx-timeline .tl-item::before{content:"";position:absolute;left:-2.06rem;top:.32rem;width:.8rem;height:.8rem;border-radius:50%;background:var(--fx-surface);border:3px solid var(--fx-accent);}
    .fxbook .fx-timeline .tl-item.future::before{border-color:var(--fx-amber);}
    .fxbook .fx-timeline .tl-when{font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:var(--fx-accent);}
    .fxbook .fx-timeline .tl-item.future .tl-when{color:var(--fx-amber);}
    .fxbook .fx-timeline h5{font-size:1rem;margin:.15rem 0 .25rem;}
    .fxbook .fx-timeline p{font-size:.93rem;margin-bottom:.2rem;}

    .fxbook .fx-acc-group{margin:1.2rem 0;}
    .fxbook details.fx-acc{border:1px solid var(--fx-line);border-radius:12px;background:var(--fx-surface);margin-bottom:.6rem;overflow:hidden;}
    .fxbook details.fx-acc summary{cursor:pointer;list-style:none;padding:.85rem 2.6rem .85rem 1.1rem;position:relative;font-weight:700;font-family:Georgia,serif;font-size:1.02rem;}
    .fxbook details.fx-acc summary::-webkit-details-marker{display:none;}
    .fxbook details.fx-acc summary::after{content:"+";position:absolute;right:1.1rem;top:.75rem;font-size:1.2rem;font-weight:800;color:var(--fx-accent);}
    .fxbook details.fx-acc[open] summary::after{content:"\\2212";}
    .fxbook details.fx-acc[open] summary{border-bottom:1px dashed var(--fx-line);}
    .fxbook details.fx-acc .fx-acc-body{padding:.5rem 1.1rem 1rem;}

    .fxbook .keypoints li{margin-bottom:.35rem;}
    .fxbook .quote-block{border-left:4px solid var(--fx-accent);padding:.4rem 0 .4rem 1.2rem;font-family:Georgia,serif;font-size:1.08rem;font-style:italic;color:var(--fx-muted);margin:1.4rem 0;}
    .fxbook .glossary-grid{display:grid;grid-template-columns:1fr;gap:0 2.2rem;}
    @media (min-width:860px){.fxbook .glossary-grid{grid-template-columns:1fr 1fr;}}
    .fxbook .glossary-grid dt{font-weight:800;color:var(--fx-accent);margin-top:.9rem;}
    .fxbook .glossary-grid dd{margin:0 0 .3rem;font-size:.94rem;}
    .fxbook .story-step{display:flex;gap:1rem;margin-bottom:1.1rem;}
    .fxbook .story-step .num{flex:0 0 2.1rem;height:2.1rem;border-radius:50%;background:var(--fx-accent);color:#fff;font-weight:800;display:flex;align-items:center;justify-content:center;font-size:.95rem;}
    .fxbook .story-step .num.warm{background:var(--fx-amber);}
    .fxbook .story-step h5{font-size:1rem;margin:0 0 .2rem;}
    .fxbook .story-step p{font-size:.94rem;margin-bottom:0;}
    .fxbook .big-tree{background:var(--fx-code-bg);border:1px solid var(--fx-line);border-radius:12px;padding:1.2rem;font-family:ui-monospace,Consolas,monospace;font-size:.8rem;line-height:1.45;overflow-x:auto;white-space:pre;}
    .fxbook mark{background:rgba(180,83,9,.3);border-radius:3px;padding:0 .12rem;}

    .fxbook .text-body-secondary{color:var(--fx-muted);}
    .fxbook .text-center{text-align:center;}
    .fxbook .fs-5{font-size:1.15rem;}
    .fxbook .fw-bold{font-weight:700;}
    .fxbook .mb-0{margin-bottom:0;}.fxbook .mb-1{margin-bottom:.25rem;}.fxbook .mb-2{margin-bottom:.5rem;}
    .fxbook .mb-3{margin-bottom:1rem;}.fxbook .mb-4{margin-bottom:1.5rem;}
    .fxbook .mt-2{margin-top:.5rem;}.fxbook .mt-4{margin-top:1.5rem;}
    .fxbook .my-1{margin-top:.25rem;margin-bottom:.25rem;}.fxbook .my-2{margin-top:.5rem;margin-bottom:.5rem;}
    .fxbook .p-3{padding:1rem;}.fxbook .rounded{border-radius:12px;}

    /* ---- theme shield ----------------------------------------------------
       Doubled class = specificity (0,2,x), which beats typical Blogspot theme
       rules such as ".post-body h2" or bare "h2/p/a/table" styling. Deliberately
       no !important and no margin overrides, so the post's own spacing system
       always stays in charge. A stubborn theme using !important can still win. */
    .fxbook.fxbook{color-scheme:light;background:var(--fx-paper);color:var(--fx-ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue","Noto Sans",Arial,sans-serif;font-size:1.02rem;line-height:1.75;}
    .fxbook.fxbook h1,.fxbook.fxbook h2,.fxbook.fxbook h3,.fxbook.fxbook h4,.fxbook.fxbook h5{color:var(--fx-ink);font-family:Georgia,"Times New Roman","Noto Serif",serif;}
    .fxbook.fxbook a{color:var(--fx-accent);text-decoration:none;}
    .fxbook.fxbook table{border-collapse:collapse;width:100%;}
    .fxbook.fxbook th,.fxbook.fxbook td{padding:.6rem .85rem;}
    .fxbook.fxbook pre,.fxbook.fxbook code{font-family:ui-monospace,"Cascadia Code",Consolas,"Courier New",monospace;}
    .fxbook.fxbook summary{cursor:pointer;font-family:Georgia,"Times New Roman","Noto Serif",serif;}
    .fxbook.fxbook img{max-width:100%;height:auto;}
    .fxbook.fxbook ul,.fxbook.fxbook ol{padding-left:1.4rem;}
    .fxbook.fxbook p,.fxbook.fxbook li,.fxbook.fxbook dt,.fxbook.fxbook dd,
    .fxbook.fxbook td,.fxbook.fxbook th,.fxbook.fxbook small{color:var(--fx-ink);}
    .fxbook.fxbook table{background:var(--fx-surface);}
    .fxbook.fxbook .fx-table th,.fxbook.fxbook .fx-table td{border:none;border-bottom:1px solid var(--fx-line);}
    .fxbook.fxbook .mathbox td{border:none;}

    /* re-assert the post's own colored text at shield-level specificity */
    .fxbook.fxbook .chapter-intro,.fxbook.fxbook .text-body-secondary,
    .fxbook.fxbook .fx-diagram figcaption,.fxbook.fxbook .fnote,
    .fxbook.fxbook .quote-block,.fxbook.fxbook .fx-stat small{color:var(--fx-muted);}
    .fxbook.fxbook .toc-part,.fxbook.fxbook .glossary-grid dt,
    .fxbook.fxbook .chapter-kicker,.fxbook.fxbook .part-label{color:var(--fx-accent);}

    @media print{
      .fxbook section.chapter{page-break-before:always;}
      .fxbook .fx-diagram,.fxbook .callout,.fxbook .wwhw .card,.fxbook .fx-table{box-shadow:none;}
    }
'''

# ---- 7. assemble -----------------------------------------------------------
out = []
out.append('<!DOCTYPE html>')
out.append('<html lang="en">')
out.append('<head>')
out.append('<meta charset="utf-8">')
out.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
out.append('<title>Fontaine AI: How an AI Model Is Designed, Trained, and Scaled</title>')
out.append('</head>')
out.append('<body style="margin:0;background:#e9e6df;">')
out.append('')
out.append('<!-- ============================================================ -->')
out.append('<!--  HOW TO PUBLISH ON BLOGGER                                    -->')
out.append('<!--  1. Open your blog, create a new post, switch to HTML view.   -->')
out.append('<!--  2. Copy EVERYTHING from the <style> tag below through the    -->')
out.append('<!--     closing </article> tag, and paste it into the post.       -->')
out.append('<!--  3. Publish. No scripts, sidebar, navbar or external files    -->')
out.append('<!--     are needed; every style is scoped under .fxbook so the    -->')
out.append('<!--     post cannot clash with your blog theme.                   -->')
out.append('<!--     If your theme strips <style> tags, paste the CSS into     -->')
out.append('<!--     Theme > Customise > Advanced > Add CSS instead.           -->')
out.append('<!-- ============================================================ -->')
out.append('')
out.append('<style>' + css + '</style>')
out.append('')
out.append('<article class="fxbook" id="top">')
out.append(content)
out.append('</article>')
out.append('')
out.append('</body>')
out.append('</html>')

io.open(DST, "w", encoding="utf-8").write("\n".join(out))
print(f"written {DST}: {len(chr(10).join(out))} bytes, accordion items converted: {n_items}, toc groups: {len(toc_groups)}")

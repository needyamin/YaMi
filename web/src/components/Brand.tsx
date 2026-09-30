/** YaMi v1.0 mascot — the pixel helper from assets/yami-icon.svg.
 *
 * Same 14px grid and palette as that file (shadow #1b4fbf, body #2f6feb,
 * mid #3d7bf0, highlight #8fd0ff) so the two images cannot drift apart. The
 * rounded background of that file is NOT repeated here: in the app the mark
 * sits on the interface surface, which brings its own. The stem and tip sit
 * above the head, so the viewBox starts a row earlier than the icon's.
 *
 * The tints are fixed rather than currentColor: they are saturated enough to
 * hold up on both the dark and the light theme. The word beside them stays in
 * currentColor, so it follows the theme like the rest of the text. */
export function Brand({ word = false }: { word?: boolean }) {
  return <span className={word ? "yami-brand full" : "yami-brand"} aria-label="YaMi">
    <svg viewBox="35 7 84 140" aria-hidden="true" stroke="none"><g shapeRendering="crispEdges" stroke="none">
<rect x="42" y="14" width="14" height="14" fill="#3d7bf0"></rect><rect x="42" y="28" width="14" height="14" fill="#3d7bf0"></rect><rect x="42" y="0" width="14" height="14" fill="#8fd0ff"></rect>
<rect x="14" y="42" width="14" height="14" fill="#2f6feb"></rect><rect x="28" y="42" width="14" height="14" fill="#2f6feb"></rect><rect x="42" y="42" width="14" height="14" fill="#2f6feb"></rect><rect x="56" y="42" width="14" height="14" fill="#2f6feb"></rect>
<rect x="0" y="56" width="14" height="14" fill="#2f6feb"></rect><rect x="14" y="56" width="14" height="14" fill="#3d7bf0"></rect><rect x="28" y="56" width="14" height="14" fill="#fff"></rect><rect x="42" y="56" width="14" height="14" fill="#3d7bf0"></rect><rect x="56" y="56" width="14" height="14" fill="#fff"></rect><rect x="70" y="56" width="14" height="14" fill="#2f6feb"></rect>
<rect x="0" y="70" width="14" height="14" fill="#2f6feb"></rect><rect x="14" y="70" width="14" height="14" fill="#3d7bf0"></rect><rect x="28" y="70" width="14" height="14" fill="#1b4fbf"></rect><rect x="42" y="70" width="14" height="14" fill="#3d7bf0"></rect><rect x="56" y="70" width="14" height="14" fill="#1b4fbf"></rect><rect x="70" y="70" width="14" height="14" fill="#2f6feb"></rect>
<rect x="0" y="84" width="14" height="14" fill="#2f6feb"></rect><rect x="14" y="84" width="14" height="14" fill="#8fd0ff"></rect><rect x="28" y="84" width="14" height="14" fill="#3d7bf0"></rect><rect x="42" y="84" width="14" height="14" fill="#1b4fbf"></rect><rect x="56" y="84" width="14" height="14" fill="#3d7bf0"></rect><rect x="70" y="84" width="14" height="14" fill="#8fd0ff"></rect>
<rect x="0" y="98" width="14" height="14" fill="#2f6feb"></rect><rect x="14" y="98" width="14" height="14" fill="#3d7bf0"></rect><rect x="28" y="98" width="14" height="14" fill="#2f6feb"></rect><rect x="42" y="98" width="14" height="14" fill="#2f6feb"></rect><rect x="56" y="98" width="14" height="14" fill="#2f6feb"></rect><rect x="70" y="98" width="14" height="14" fill="#3d7bf0"></rect>
<rect x="14" y="112" width="14" height="14" fill="#2f6feb"></rect><rect x="28" y="112" width="14" height="14" fill="#2f6feb"></rect><rect x="42" y="112" width="14" height="14" fill="#2f6feb"></rect><rect x="56" y="112" width="14" height="14" fill="#2f6feb"></rect>
<rect x="14" y="126" width="14" height="14" fill="#1b4fbf"></rect><rect x="28" y="126" width="14" height="14" fill="#1b4fbf"></rect><rect x="42" y="126" width="14" height="14" fill="#1b4fbf"></rect><rect x="56" y="126" width="14" height="14" fill="#1b4fbf"></rect>
</g></svg>
    {word && <span className="yami-word">YaMi v1.0</span>}
  </span>
}

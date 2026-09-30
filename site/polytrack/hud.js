// The car's speed as shown on PolyTrack's on-screen display (km/h). Used by
// the autopilot teacher and by the stuck helper, never by the fly itself.
export function hudSpeed() {
  const text = document.getElementById("ui")?.innerText ?? "";
  const match = text.match(/(\d+)\s*km\/h/);
  return match ? Number(match[1]) : 0;
}

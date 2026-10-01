// Read PolyTrack's on-screen display. Used for the stuck helper and for
// automatic rewards (checkpoints, crashes), never as input to the fly.
const hudText = () => document.getElementById("ui")?.innerText ?? "";

// The car's speed in km/h.
export function hudSpeed() {
  const match = hudText().match(/(\d+)\s*km\/h/);
  return match ? Number(match[1]) : 0;
}

// Checkpoints collected so far in this race, or null if not shown.
export function hudCheckpoint() {
  const match = hudText().match(/(\d+)\s*\/\s*(\d+)/);
  return match ? Number(match[1]) : null;
}

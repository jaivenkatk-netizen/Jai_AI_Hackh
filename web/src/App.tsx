// Shell. Holds the two pieces of cross-screen state that make the demo click
// path work: the hub filter (set from the Command Center map, respected on
// every screen) and the selected asset (which switches to the Asset Twin).

import { useState } from "react";

import AssetTwin from "./screens/AssetTwin";
import CommandCenter from "./screens/CommandCenter";
import InventoryPlanner from "./screens/InventoryPlanner";
import PartsDemand from "./screens/PartsDemand";

type Screen = "command" | "parts" | "planner" | "asset";

const NAV: { key: Screen; label: string; hint: string }[] = [
  { key: "command", label: "Command Center", hint: "Fleet state and the urgent queue" },
  { key: "parts", label: "Parts Demand", hint: "Failures translated into part numbers" },
  { key: "planner", label: "Inventory Planner", hint: "Shortages turned into approvable actions" },
];

export default function App() {
  const [screen, setScreen] = useState<Screen>("command");
  const [hub, setHub] = useState<string | null>(null);
  const [assetId, setAssetId] = useState<string | null>(null);

  const openAsset = (id: string) => {
    setAssetId(id);
    setScreen("asset");
  };

  return (
    <div className="min-h-screen">
      <header
        className="sticky top-0 z-10 px-4 py-2"
        style={{ background: "var(--surface)", borderBottom: "1px solid var(--line)" }}
      >
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex items-baseline gap-2">
            <span className="text-[14px] font-semibold tracking-tight">ServiceSense</span>
            <span className="text-[11px]" style={{ color: "var(--muted)" }}>
              A01 &middot; maintenance to inventory
            </span>
          </div>
          <nav className="flex gap-1">
            {NAV.map((n) => (
              <button
                key={n.key}
                type="button"
                title={n.hint}
                onClick={() => setScreen(n.key)}
                className="rounded px-2.5 py-1 text-[12px]"
                style={{
                  background: screen === n.key ? "var(--plane)" : "transparent",
                  fontWeight: screen === n.key ? 600 : 400,
                  boxShadow: screen === n.key ? "inset 0 -2px 0 0 var(--ink)" : undefined,
                }}
              >
                {n.label}
              </button>
            ))}
            {screen === "asset" && assetId && (
              <span
                className="num rounded px-2.5 py-1 text-[12px] font-semibold"
                style={{ background: "var(--plane)", boxShadow: "inset 0 -2px 0 0 var(--ink)" }}
              >
                {assetId}
              </span>
            )}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] p-4">
        {screen === "command" && (
          <CommandCenter
            hub={hub}
            setHub={setHub}
            onOpenAsset={openAsset}
            onGoto={(s) => setScreen(s)}
          />
        )}
        {screen === "parts" && <PartsDemand hub={hub} setHub={setHub} onOpenAsset={openAsset} />}
        {screen === "planner" && <InventoryPlanner hub={hub} setHub={setHub} />}
        {screen === "asset" && assetId && (
          <AssetTwin assetId={assetId} onBack={() => setScreen("command")} />
        )}
      </main>
    </div>
  );
}

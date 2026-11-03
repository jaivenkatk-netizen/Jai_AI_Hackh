// The digital twin, in three dimensions: the actual refrigerant circuit of
// this machine, with each part shaded by its predicted health and the
// condenser fan turning.
//
// Two rules this view inherits from the rest of the app, and one it adds.
//
//   Colour is not free. Red / amber / blue already mean urgent / planned /
//   forecast on every other screen, so health here rides a single-hue
//   sequential ramp - a degrading part darkens, it does not change colour.
//   Using green-to-red would have made a healthy compressor read as a
//   different kind of object from an urgent one.
//
//   Geometry is not decoration. The layout is the thermodynamic cycle in
//   order - compressor, condenser, expansion valve, evaporator, back to
//   suction - so the picture carries the same information as the P-h diagram
//   a service engineer already has in their head. The flow arrows follow the
//   refrigerant, not an arbitrary circuit.
//
//   New here: shading encodes a prediction, so the view has to say so. The
//   legend states what drives the colour and the panel beneath carries the
//   inferred charge separately, labelled with whether it survived validation.

import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Group, Mesh } from "three";

import { titleCase } from "../lib/fmt";

// The same sequential ramp the 2D schematic uses, dark = degraded.
const RAMP = ["#0b2f5c", "#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"];

const healthColor = (h: number | undefined): string => {
  if (h === undefined || Number.isNaN(h)) return "#5b6572";
  const i = Math.min(Math.max(Math.floor(h * RAMP.length), 0), RAMP.length - 1);
  return RAMP[i];
};

/** One labelled, clickable part of the circuit. */
type Part = {
  key: string;
  label: string;
  /** position / size in cycle space */
  pos: [number, number, number];
  size: [number, number, number];
  kind: "box" | "cylinder" | "coil" | "valve";
};

// Laid out as the cycle runs: compression up the left, rejection across the
// top, expansion down the right, evaporation back along the bottom.
const PARTS: Part[] = [
  { key: "compressor", label: "Compressor", pos: [-1.5, -0.15, 0], size: [0.52, 1.0, 0.52], kind: "cylinder" },
  { key: "condenser_coil", label: "Condenser coil", pos: [0, 1.05, 0], size: [1.9, 0.62, 0.78], kind: "coil" },
  { key: "condenser_fan_motor", label: "Condenser fan", pos: [0, 1.72, 0], size: [0.34, 0.34, 0.34], kind: "box" },
  { key: "eev", label: "EEV / TXV", pos: [1.5, 0.1, 0], size: [0.34, 0.46, 0.34], kind: "valve" },
  { key: "evaporator", label: "Evaporator", pos: [0, -1.15, 0], size: [1.9, 0.62, 0.78], kind: "coil" },
  { key: "evap_blower_motor", label: "Evaporator blower", pos: [0, -1.82, 0], size: [0.34, 0.34, 0.34], kind: "box" },
];

// Pipe runs, as straight segments between cycle stations.
const PIPES: { from: [number, number, number]; to: [number, number, number]; hot: boolean }[] = [
  { from: [-1.5, 0.4, 0], to: [-1.5, 1.05, 0], hot: true }, // discharge
  { from: [-1.5, 1.05, 0], to: [-0.95, 1.05, 0], hot: true },
  { from: [0.95, 1.05, 0], to: [1.5, 1.05, 0], hot: true }, // liquid line
  { from: [1.5, 1.05, 0], to: [1.5, 0.35, 0], hot: true },
  { from: [1.5, -0.15, 0], to: [1.5, -1.15, 0], hot: false }, // to evaporator
  { from: [1.5, -1.15, 0], to: [0.95, -1.15, 0], hot: false },
  { from: [-0.95, -1.15, 0], to: [-1.5, -1.15, 0], hot: false }, // suction
  { from: [-1.5, -1.15, 0], to: [-1.5, -0.68, 0], hot: false },
];

function Pipe({ from, to, hot }: { from: [number, number, number]; to: [number, number, number]; hot: boolean }) {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy) || 0.001;
  const mid: [number, number, number] = [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2, from[2]];
  // Rotate the cylinder (default +Y) onto the segment direction
  const rot: [number, number, number] = [0, 0, Math.atan2(dy, dx) - Math.PI / 2];
  return (
    <mesh position={mid} rotation={rot}>
      <cylinderGeometry args={[0.055, 0.055, len, 12]} />
      {/* Warm metal on the high-pressure side, cool on the suction side.
          This is the one place hue carries a physical meaning rather than a
          horizon, and it is never used for health. */}
      <meshStandardMaterial
        color={hot ? "#b08968" : "#8d9db6"}
        metalness={0.65}
        roughness={0.35}
      />
    </mesh>
  );
}

/** Fin stack, so a coil reads as a coil rather than a slab. */
function Coil({ size, color }: { size: [number, number, number]; color: string }) {
  const [w, h, d] = size;
  const n = 13;
  return (
    <group>
      {Array.from({ length: n }, (_, i) => (
        <mesh key={i} position={[-w / 2 + (w * (i + 0.5)) / n, 0, 0]}>
          <boxGeometry args={[(w / n) * 0.45, h, d]} />
          <meshStandardMaterial color={color} metalness={0.25} roughness={0.6} />
        </mesh>
      ))}
      {/* header tubes top and bottom */}
      {[h / 2, -h / 2].map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.05, 0.05, w, 10]} />
          <meshStandardMaterial color="#8d9db6" metalness={0.7} roughness={0.3} />
        </mesh>
      ))}
    </group>
  );
}

/** Four blades that actually turn, at a rate set by the duty fraction. */
function Fan({ color, speed }: { color: string; speed: number }) {
  const ref = useRef<Group>(null);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * speed;
  });
  return (
    <group ref={ref}>
      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} rotation={[0, (i * Math.PI) / 2, 0.42]} position={[0, 0, 0]}>
          <boxGeometry args={[0.46, 0.028, 0.15]} />
          <meshStandardMaterial color={color} metalness={0.4} roughness={0.5} />
        </mesh>
      ))}
      <mesh>
        <sphereGeometry args={[0.075, 14, 14]} />
        <meshStandardMaterial color="#5b6572" metalness={0.6} roughness={0.4} />
      </mesh>
    </group>
  );
}

function PartMesh({
  part,
  health,
  selected,
  onSelect,
  onHover,
  fanSpeed,
}: {
  part: Part;
  health: number | undefined;
  selected: boolean;
  onSelect: () => void;
  onHover: (k: string | null) => void;
  fanSpeed: number;
}) {
  const ref = useRef<Mesh>(null);
  const color = healthColor(health);
  const [w, h, d] = part.size;

  // A selected part lifts slightly toward the viewer. Scaling it would have
  // changed the geometry the view is meant to be honest about.
  const z = part.pos[2] + (selected ? 0.22 : 0);

  const body = () => {
    switch (part.kind) {
      case "cylinder":
        return (
          <mesh ref={ref}>
            <cylinderGeometry args={[w / 2, w / 2, h, 22]} />
            <meshStandardMaterial color={color} metalness={0.45} roughness={0.45} />
          </mesh>
        );
      case "coil":
        return <Coil size={part.size} color={color} />;
      case "valve":
        return (
          <mesh ref={ref}>
            <coneGeometry args={[w / 1.5, h, 4]} />
            <meshStandardMaterial color={color} metalness={0.5} roughness={0.4} />
          </mesh>
        );
      default:
        return part.key.includes("fan") || part.key.includes("blower") ? (
          <Fan color={color} speed={fanSpeed} />
        ) : (
          <mesh ref={ref}>
            <boxGeometry args={[w, h, d]} />
            <meshStandardMaterial color={color} metalness={0.3} roughness={0.55} />
          </mesh>
        );
    }
  };

  return (
    <group
      position={[part.pos[0], part.pos[1], z]}
      onPointerDown={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(part.key);
      }}
      onPointerOut={() => onHover(null)}
    >
      {body()}
      {selected && (
        // Selection ring, drawn in the ink colour so it does not compete with
        // the health ramp.
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
          <torusGeometry args={[Math.max(w, h) * 0.78, 0.018, 8, 40]} />
          <meshStandardMaterial color="#e8edf5" metalness={0.2} roughness={0.8} />
        </mesh>
      )}
    </group>
  );
}

function Scene({
  health,
  selected,
  onSelect,
  onHover,
  fanSpeed,
  spin,
  auto,
}: {
  health: Record<string, number>;
  selected: string | null;
  onSelect: (k: string) => void;
  onHover: (k: string | null) => void;
  fanSpeed: number;
  spin: React.RefObject<number>;
  auto: React.RefObject<boolean>;
}) {
  const g = useRef<Group>(null);
  useFrame((_, dt) => {
    if (auto.current) spin.current += dt * 0.28;
    if (g.current) g.current.rotation.y = spin.current;
  });
  return (
    <>
      <ambientLight intensity={0.62} />
      <directionalLight position={[4, 6, 5]} intensity={1.05} />
      <directionalLight position={[-5, -2, -4]} intensity={0.3} />
      <group ref={g}>
        {PIPES.map((p, i) => (
          <Pipe key={i} {...p} />
        ))}
        {/* Only the parts this machine actually carries. A chiller has no
            evaporator blower, so drawing one would be inventing hardware. */}
        {PARTS.filter((p) => health[p.key] !== undefined).map((p) => (
          <PartMesh
            key={p.key}
            part={p}
            health={health[p.key]}
            selected={selected === p.key}
            onSelect={() => onSelect(p.key)}
            onHover={onHover}
            fanSpeed={fanSpeed}
          />
        ))}
      </group>
    </>
  );
}

/** Is WebGL actually available in this browser? */
function webglOk(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

function Twin3DImpl({
  health,
  selectedType,
  onSelectType,
  dutyFrac = 1,
  height = 300,
  fallback,
}: {
  health: Record<string, number>;
  selectedType?: string | null;
  onSelectType?: (componentType: string) => void;
  dutyFrac?: number;
  height?: number;
  /** Rendered instead of the canvas where WebGL is unavailable. */
  fallback?: ReactNode;
}) {
  // Checked once on mount rather than at module load, because a module-level
  // probe runs during SSR and prerender where there is no canvas at all.
  // A demo machine without a working GPU should still show the circuit.
  const [gl, setGl] = useState<boolean | null>(null);
  useEffect(() => setGl(webglOk()), []);

  // The rotation lives in refs, not in state, and is applied inside the
  // render loop. Driving it through setState at 60 fps re-rendered this
  // whole screen - the component table, the charts, the what-if panel - once
  // per frame, which is the classic way an otherwise cheap 3D view makes an
  // app feel broken. Nothing here needs React to know the current angle.
  const spin = useRef(0.5);
  const auto = useRef(true);
  const [autoShown, setAutoShown] = useState(true); // for the hint and button only
  const [hover, setHover] = useState<string | null>(null);
  const drag = useRef<{ x: number; from: number } | null>(null);
  const [grabbing, setGrabbing] = useState(false);

  const stopAuto = () => {
    auto.current = false;
    setAutoShown(false);
  };
  const resumeAuto = () => {
    auto.current = true;
    setAutoShown(true);
  };

  const fanSpeed = useMemo(() => {
    const h = health["condenser_fan_motor"];
    // A degraded fan turns visibly slower; duty scales it too, so a cycling
    // machine does not look like one running flat out.
    return Math.max(0.35, (h ?? 1) * 5.5 * Math.max(dutyFrac, 0.15));
  }, [health, dutyFrac]);

  const label = hover ?? selectedType ?? null;
  const labelPart = PARTS.find((p) => p.key === label);
  const labelHealth = label ? health[label] : undefined;

  // An air handler has no refrigerant circuit at all, so there is no cycle to
  // draw. Say that rather than render an empty box.
  const present = PARTS.filter((p) => health[p.key] !== undefined).length;
  if (present < 2) {
    return (
      <div
        className="flex items-center justify-center rounded px-3 text-center text-[11px]"
        style={{ height, background: "var(--plane)", color: "var(--ink-2)" }}
      >
        This machine has no refrigerant circuit, so there is no vapour-compression cycle to
        show. Its components are listed beside this panel.
      </div>
    );
  }

  if (gl === false && fallback) {
    return (
      <div>
        {fallback}
        <p className="mt-2 text-[10px]" style={{ color: "var(--muted)" }}>
          Showing the flat schematic: this browser has no WebGL, so the three-dimensional view
          cannot render. The same health shading applies.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div
        style={{ height, touchAction: "none", cursor: grabbing ? "grabbing" : "grab" }}
        className="relative overflow-hidden rounded"
        onPointerDown={(e) => {
          stopAuto();
          setGrabbing(true);
          drag.current = { x: e.clientX, from: spin.current };
        }}
        onPointerMove={(e) => {
          // Writes the ref; the render loop picks it up on the next frame.
          if (drag.current) spin.current = drag.current.from + (e.clientX - drag.current.x) * 0.011;
        }}
        onPointerUp={() => {
          drag.current = null;
          setGrabbing(false);
        }}
        onPointerLeave={() => {
          drag.current = null;
          setGrabbing(false);
          setHover(null);
        }}
      >
        <Canvas
          camera={{ position: [0, 0.4, 6.4], fov: 42 }}
          // The canvas is decorative relative to the tables beside it, so it
          // does not need to be in the tab order; every part it shows is also
          // reachable from the component list.
          tabIndex={-1}
          dpr={[1, 1.75]}
          style={{ background: "transparent" }}
        >
          <Scene
            health={health}
            selected={selectedType ?? null}
            onSelect={(k) => onSelectType?.(k)}
            onHover={setHover}
            fanSpeed={fanSpeed}
            spin={spin}
            auto={auto}
          />
        </Canvas>

        {labelPart && (
          <div
            className="pointer-events-none absolute left-2 top-2 rounded px-2 py-1 text-[11px]"
            style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
          >
            <span className="font-medium">{labelPart.label}</span>
            {labelHealth !== undefined && (
              <span className="num ml-1.5" style={{ color: "var(--ink-2)" }}>
                health {(labelHealth * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}

        {autoShown && (
          <div
            className="pointer-events-none absolute bottom-2 right-2 text-[10px]"
            style={{ color: "var(--muted)" }}
          >
            drag to rotate
          </div>
        )}
      </div>

      {/* The legend states what the colour means and where it comes from,
          because shading here encodes a prediction rather than a reading. */}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px]">
        <span style={{ color: "var(--ink-2)" }}>Shaded by predicted health:</span>
        <span className="flex items-center gap-1">
          <span>degraded</span>
          <span className="flex">
            {RAMP.map((c) => (
              <span key={c} className="inline-block h-2.5 w-4" style={{ background: c }} />
            ))}
          </span>
          <span>healthy</span>
        </span>
        <span style={{ color: "var(--muted)" }}>
          from each part's P50 time-to-maintenance, not a sensor reading
        </span>
      </div>
      {!autoShown && (
        <button
          type="button"
          onClick={resumeAuto}
          className="mt-1 rounded px-1.5 py-0.5 text-[10px]"
          style={{ border: "1px solid var(--line)" }}
        >
          resume rotation
        </button>
      )}
      {hover && onSelectType && (
        <p className="mt-1 text-[10px]" style={{ color: "var(--muted)" }}>
          Click {titleCase(hover)} to select it in the list
        </p>
      )}
    </div>
  );
}

// three.js is about 870 kB of the bundle and only the Asset Twin screen needs
// it, so it loads on demand rather than on first paint. The Command Center is
// the entry point for the demo and should not wait for a 3D engine.
export default Twin3DImpl;

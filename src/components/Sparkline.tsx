import { useId } from "react";

export interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillColor?: string;
  min?: number;
  max?: number;
  unit?: string;
  label?: string;
  currentValue?: number | string;
  formatValue?: (val: number) => string;
  className?: string;
  showMinMax?: boolean;
}

export function Sparkline({
  data,
  width = 240,
  height = 44,
  color = "#818cf8", // indigo-400
  fillColor,
  min,
  max,
  unit = "",
  label,
  currentValue,
  formatValue,
  className = "",
  showMinMax = true,
}: SparklineProps) {
  const gradId = useId();
  const paddingX = 4;
  const paddingY = 4;

  const validData = data.filter((v) => Number.isFinite(v));
  const hasPoints = validData.length > 0;

  const computedMin = min !== undefined ? min : hasPoints ? Math.min(...validData) : 0;
  const computedMax = max !== undefined ? max : hasPoints ? Math.max(...validData) : 100;
  const range = computedMax - computedMin || 1;

  const latestVal =
    currentValue !== undefined
      ? currentValue
      : validData.length > 0
        ? formatValue
          ? formatValue(validData[validData.length - 1])
          : validData[validData.length - 1].toFixed(1)
        : "—";

  // Build SVG points
  let polylinePoints = "";
  let polygonPoints = "";
  let lastX = paddingX;
  let lastY = height / 2;

  if (validData.length > 1) {
    const pts = validData.map((val, idx) => {
      const x = paddingX + (idx / (validData.length - 1)) * (width - 2 * paddingX);
      const clamped = Math.max(computedMin, Math.min(computedMax, val));
      const y = paddingY + (1 - (clamped - computedMin) / range) * (height - 2 * paddingY);
      return [x, y] as [number, number];
    });

    polylinePoints = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const [firstX] = pts[0];
    const [endX, endY] = pts[pts.length - 1];
    lastX = endX;
    lastY = endY;
    const bottomY = height - paddingY;
    polygonPoints = `${polylinePoints} ${endX.toFixed(1)},${bottomY} ${firstX.toFixed(1)},${bottomY}`;
  } else if (validData.length === 1) {
    const clamped = Math.max(computedMin, Math.min(computedMax, validData[0]));
    lastY = paddingY + (1 - (clamped - computedMin) / range) * (height - 2 * paddingY);
    lastX = width / 2;
  }

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {(label || latestVal !== undefined) && (
        <div className="flex items-center justify-between text-xs">
          {label && <span className="font-medium text-slate-400">{label}</span>}
          <span className="font-mono font-semibold" style={{ color }}>
            {latestVal}
            {unit && <span className="text-[10px] text-slate-400 ml-0.5">{unit}</span>}
          </span>
        </div>
      )}

      <div className="relative overflow-hidden rounded bg-black/20 border border-edge/40 px-1 py-0.5">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto block"
          style={{ maxHeight: height }}
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={fillColor || color} stopOpacity="0.35" />
              <stop offset="100%" stopColor={fillColor || color} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Reference grid line at 50% */}
          <line
            x1={paddingX}
            y1={height / 2}
            x2={width - paddingX}
            y2={height / 2}
            stroke="#334155"
            strokeDasharray="2 4"
            strokeWidth="0.75"
            opacity="0.4"
          />

          {validData.length > 1 ? (
            <>
              <polygon points={polygonPoints} fill={`url(#${gradId})`} />
              <polyline
                points={polylinePoints}
                fill="none"
                stroke={color}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <circle cx={lastX} cy={lastY} r="2.5" fill={color} />
            </>
          ) : validData.length === 1 ? (
            <>
              <line
                x1={paddingX}
                y1={lastY}
                x2={width - paddingX}
                y2={lastY}
                stroke={color}
                strokeWidth="1.5"
                strokeDasharray="3 3"
              />
              <circle cx={lastX} cy={lastY} r="3" fill={color} />
            </>
          ) : (
            <line
              x1={paddingX}
              y1={height / 2}
              x2={width - paddingX}
              y2={height / 2}
              stroke="#475569"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
          )}
        </svg>

        {showMinMax && (
          <div className="flex justify-between items-center px-0.5 mt-0.5 text-[9px] text-slate-500 font-mono">
            <span>
              {formatValue ? formatValue(computedMin) : computedMin.toFixed(0)}
              {unit}
            </span>
            <span className="text-[8px] text-slate-600">5m</span>
            <span>
              {formatValue ? formatValue(computedMax) : computedMax.toFixed(0)}
              {unit}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

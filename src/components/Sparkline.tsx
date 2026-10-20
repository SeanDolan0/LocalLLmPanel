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
  // A string currentValue is already formatted by the caller (and often
  // includes its own unit), so only append `unit` for numeric values.
  const showUnit = unit && (currentValue === undefined || typeof currentValue === "number");

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
            {showUnit && <span className="text-[10px] text-slate-400 ml-0.5">{unit}</span>}
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

export interface SparklineSeries {
  label: string;
  data: number[];
  color: string;
}

export interface MultiSparklineProps {
  series: SparklineSeries[];
  width?: number;
  height?: number;
  unit?: string;
  label?: string;
  min?: number;
  max?: number;
  formatValue?: (val: number) => string;
  className?: string;
  showMinMax?: boolean;
}

/** A compact multi-line chart with a shared y-axis and an inline legend. */
export function MultiSparkline({
  series,
  width = 280,
  height = 48,
  unit = "",
  label,
  min,
  max,
  formatValue,
  className = "",
  showMinMax = true,
}: MultiSparklineProps) {
  const paddingX = 4;
  const paddingY = 6;
  const allValues = series.flatMap((s) => s.data).filter((v) => Number.isFinite(v));
  const hasPoints = allValues.length > 0;
  const computedMin = min !== undefined ? min : hasPoints ? Math.min(...allValues) : 0;
  const computedMax = max !== undefined ? max : hasPoints ? Math.max(...allValues) : 1;
  const range = computedMax - computedMin || 1;
  const fmt = formatValue ?? ((v: number) => v.toFixed(1));

  const lines = series.map((s) => {
    const valid = s.data.filter((v) => Number.isFinite(v));
    const points = valid.map((val, idx) => {
      const x =
        paddingX +
        (valid.length <= 1 ? 0 : (idx / (valid.length - 1)) * (width - 2 * paddingX));
      const clamped = Math.max(computedMin, Math.min(computedMax, val));
      const y = paddingY + (1 - (clamped - computedMin) / range) * (height - 2 * paddingY);
      return [x, y] as const;
    });
    return {
      ...s,
      points,
      last: valid.length > 0 ? valid[valid.length - 1] : null,
    };
  });

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs">
        {label ? <span className="font-medium text-slate-400">{label}</span> : <span />}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
          {lines.map((line) => (
            <span key={line.label} className="inline-flex items-center gap-1 font-mono">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: line.color }} />
              <span className="text-[10px] text-slate-500">{line.label}</span>
              <span className="text-[10px] font-semibold" style={{ color: line.color }}>
                {line.last != null ? fmt(line.last) : "—"}
                {unit && <span className="ml-0.5 text-[9px] font-normal text-slate-500">{unit}</span>}
              </span>
            </span>
          ))}
        </div>
      </div>

      <div className="relative overflow-hidden rounded bg-black/20 border border-edge/40 px-1 py-0.5">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto block"
          style={{ maxHeight: height }}
          preserveAspectRatio="none"
        >
          {[0.25, 0.5, 0.75].map((fraction) => (
            <line
              key={fraction}
              x1={paddingX}
              y1={height * fraction}
              x2={width - paddingX}
              y2={height * fraction}
              stroke="#334155"
              strokeDasharray="2 4"
              strokeWidth="0.75"
              opacity="0.35"
            />
          ))}

          {hasPoints ? (
            lines.map((line) => (
              <g key={line.label}>
                {line.points.length > 1 && (
                  <polyline
                    points={line.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")}
                    fill="none"
                    stroke={line.color}
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                )}
                {line.points.length > 0 && (
                  <circle
                    cx={line.points[line.points.length - 1][0]}
                    cy={line.points[line.points.length - 1][1]}
                    r="2.5"
                    fill={line.color}
                  />
                )}
              </g>
            ))
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
              {fmt(computedMin)}
              {unit}
            </span>
            <span className="text-[8px] text-slate-600">5m</span>
            <span>
              {fmt(computedMax)}
              {unit}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

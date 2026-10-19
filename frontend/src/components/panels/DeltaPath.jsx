import { Area, AreaChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from "recharts";

/** Current window's delta (bps) vs elapsed seconds — the path, as a time series. */
export default function DeltaPath({ tick, accent = "#38bdf8" }) {
  const x = tick?.x ?? [];
  const y = tick?.y ?? [];
  const data = x.map((t, i) => ({ t, delta: y[i] }));
  if (!data.length) return <div className="grid h-full place-items-center text-[10px] text-muted">no data</div>;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
        <defs>
          <linearGradient id="dGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={accent} stopOpacity={0.3} />
            <stop offset="95%" stopColor={accent} stopOpacity={0} />
          </linearGradient>
        </defs>
        <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
        <XAxis dataKey="t" hide />
        <YAxis hide domain={["auto", "auto"]} />
        <Area type="monotone" dataKey="delta" stroke={accent} strokeWidth={1.5} fill="url(#dGrad)" dot={false} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

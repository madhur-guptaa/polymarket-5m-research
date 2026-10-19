import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

/** 1-hour average-price trend (from the `macro` message). */
export default function MacroTrend({ data = [], accent = "#38bdf8" }) {
  if (!data.length) return <div className="grid h-full place-items-center text-[10px] text-muted">no trend</div>;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
        <defs>
          <linearGradient id="mGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={accent} stopOpacity={0.25} />
            <stop offset="95%" stopColor={accent} stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="minute_time" hide />
        <YAxis hide domain={["auto", "auto"]} />
        <Tooltip
          contentStyle={{ background: "#0d1117", border: "1px solid #1e2937", borderRadius: 6, fontSize: 10, fontFamily: "monospace" }}
          labelFormatter={() => ""}
        />
        <Area type="monotone" dataKey="avg_price" stroke={accent} strokeWidth={1.5} fill="url(#mGrad)" dot={false} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

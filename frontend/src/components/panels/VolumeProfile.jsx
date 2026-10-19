import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";

/** 10-second volume bars over the last 5 minutes (from the `micro` message). */
export default function VolumeProfile({ data = [], accent = "#38bdf8" }) {
  if (!data.length) return <div className="grid h-full place-items-center text-[10px] text-muted">no volume</div>;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
        <XAxis dataKey="bucket_time" hide />
        <Tooltip
          cursor={{ fill: "#ffffff08" }}
          contentStyle={{ background: "#0d1117", border: "1px solid #1e2937", borderRadius: 6, fontSize: 10, fontFamily: "monospace" }}
          labelFormatter={() => ""}
        />
        <Bar dataKey="volume" fill={accent} fillOpacity={0.35} stroke={accent} strokeWidth={0.5} radius={[2, 2, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

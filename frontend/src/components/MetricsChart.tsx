import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface MetricsChartSeries {
  dataKey: string;
  name: string;
  color: string;
  strokeWidth?: number;
  strokeDasharray?: string;
  type?: "monotone" | "linear" | "natural" | "step" | "stepAfter" | "stepBefore";
  area?: boolean;
}

interface MetricsChartProps {
  title: string;
  data: Array<Record<string, number>>;
  series: MetricsChartSeries[];
  yAxisLabel?: string;
  syncId?: string;
  hideLegend?: boolean;
}

export const MetricsChart = ({ title, data, series, yAxisLabel, syncId, hideLegend = false }: MetricsChartProps) => {
  return (
    <Card className="shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/30 transition-colors">
      <CardHeader>
        <CardTitle className="text-lg font-semibold bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data} syncId={syncId} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
            <defs>
              {series
                .filter((s) => s.area)
                .map((s, idx) => (
                  <linearGradient key={`${title}-gradient-${idx}`} id={`gradient-${title}-${idx}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={s.color} stopOpacity={0.25} />
                    <stop offset="95%" stopColor={s.color} stopOpacity={0} />
                  </linearGradient>
                ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.25)" />
            <XAxis
              dataKey="episode"
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
              label={
                yAxisLabel
                  ? { value: yAxisLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: "hsl(var(--muted-foreground))" }
                  : undefined
              }
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "var(--radius)",
                backdropFilter: "blur(12px)",
              }}
              formatter={(value: number | string, name: string) => [Number(value).toFixed(3), name]}
            />
            {!hideLegend && <Legend />}
            {series.map((s, idx) => (
              <Line
                key={`${title}-${s.dataKey}`}
                type={s.type ?? "monotone"}
                dataKey={s.dataKey}
                name={s.name}
                stroke={s.color}
                strokeWidth={s.strokeWidth ?? (idx === 0 ? 2.4 : 1.6)}
                strokeDasharray={s.strokeDasharray}
                dot={false}
                isAnimationActive={false}
                fill={s.area ? `url(#gradient-${title}-${idx})` : undefined}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

import { Card } from "@/components/ui/card";
import { useMemo } from "react";

interface GridEntity {
  type: "drone" | "victim-rescued" | "victim-left";
  x: number;
  y: number;
}

interface DroneGridProps {
  gridSize: number;
  entities: GridEntity[];
  obstacles?: Array<{ x: number; y: number }> | number[][];
  editable?: boolean;
  onCellClick?: (x: number, y: number) => void;
}

const keyOf = (x: number, y: number) => `${x}-${y}`;

export const DroneGrid = ({ gridSize, entities, obstacles, editable = false, onCellClick }: DroneGridProps) => {
  const obstacleSet = useMemo(() => {
    if (!obstacles) return new Set<string>();
    return new Set(
      obstacles.map((entry) => {
        const [ox, oy] = Array.isArray(entry) ? entry : [entry.x, entry.y];
        return keyOf(ox, oy);
      }),
    );
  }, [obstacles]);

  const entityKey = (x: number, y: number) => entities.find((e) => e.x === x && e.y === y);

  return (
    <Card className="p-6 shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50">
      <div 
        className="grid gap-1.5 rounded-xl p-5 relative overflow-hidden"
        style={{
          gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))`,
          background: 'linear-gradient(135deg, hsl(var(--secondary) / 0.4), hsl(var(--secondary) / 0.2))',
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5" />
        {Array.from({ length: gridSize * gridSize }).map((_, idx) => {
          const x = idx % gridSize;
          const y = Math.floor(idx / gridSize);
          const entity = entityKey(x, y);
          const obstacle = obstacleSet.has(keyOf(x, y));

          return (
            <button
              key={`${x}-${y}`}
              type="button"
              onClick={() => onCellClick?.(x, y)}
              className={`aspect-square border border-border/30 bg-background/40 backdrop-blur-sm rounded-md relative flex items-center justify-center transition-all duration-300 group ${
                editable ? "cursor-pointer hover:border-accent hover:bg-accent/10" : "hover:border-primary/50 hover:bg-primary/5"
              } ${obstacle ? "bg-muted/60 border-muted/80" : ""}`}
            >
              {obstacle && (
                <div className="absolute inset-1 rounded-md bg-border/60 backdrop-blur-sm" />
              )}
              {entity?.type === "drone" && (
                <>
                  <div className="absolute inset-0 rounded-md bg-primary/20 blur-md animate-pulse" />
                  <div className="relative w-4 h-4 rounded-full bg-gradient-to-br from-primary to-blue-400 shadow-[0_0_20px_rgba(96,165,250,0.5)] animate-float" />
                </>
              )}
              {entity?.type === "victim-rescued" && (
                <>
                  <div className="absolute inset-0 rounded-md bg-chart-yellow/10 blur-sm" />
                  <div className="relative w-3 h-3 rounded-sm bg-gradient-to-br from-chart-yellow to-amber-400 shadow-[0_0_15px_rgba(250,204,21,0.4)]" />
                </>
              )}
              {entity?.type === "victim-left" && (
                <>
                  <div className="absolute inset-0 rounded-md bg-destructive/20 blur-sm animate-glow-pulse" />
                  <div className="relative w-3 h-3 rounded-sm bg-gradient-to-br from-destructive to-red-500 shadow-[0_0_15px_rgba(239,68,68,0.5)] animate-pulse" />
                </>
              )}
            </button>
          );
        })}
      </div>
    </Card>
  );
};

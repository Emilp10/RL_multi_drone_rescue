import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Play, Pause, RotateCcw } from "lucide-react";

interface ControlPanelProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  onReset: () => void;
  speed: number;
  onSpeedChange: (value: number[]) => void;
  agents: number;
  onAgentsChange: (value: string) => void;
  isGreedy: boolean;
  onGreedyChange: (checked: boolean) => void;
}

export const ControlPanel = ({
  isPlaying,
  onPlayPause,
  onReset,
  speed,
  onSpeedChange,
  agents,
  onAgentsChange,
  isGreedy,
  onGreedyChange,
}: ControlPanelProps) => {
  return (
    <Card className="shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50">
      <CardContent className="p-6">
        <div className="space-y-6">
          <div className="flex gap-2">
            <Button
              onClick={onPlayPause}
              variant={isPlaying ? "secondary" : "default"}
              className="flex-1 bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-[0_0_20px_rgba(96,165,250,0.3)]"
            >
              {isPlaying ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
              {isPlaying ? "Pause" : "Play"}
            </Button>
            <Button onClick={onReset} variant="outline" className="border-border/50 hover:border-primary/30 hover:bg-primary/5">
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Agents: {agents}</label>
            <Select value={agents.toString()} onValueChange={onAgentsChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">1 Agent</SelectItem>
                <SelectItem value="3">3 Agents</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Speed</label>
              <span className="text-sm text-muted-foreground">{speed} ms</span>
            </div>
            <Slider
              value={[speed]}
              onValueChange={onSpeedChange}
              min={20}
              max={500}
              step={20}
              className="w-full"
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Greedy (ε=0)</label>
            <Button
              variant={isGreedy ? "default" : "outline"}
              size="sm"
              onClick={() => onGreedyChange(!isGreedy)}
            >
              {isGreedy ? "On" : "Off"}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

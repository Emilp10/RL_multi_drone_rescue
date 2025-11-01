import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DroneGrid } from "@/components/DroneGrid";
import { MetricsChart } from "@/components/MetricsChart";
import { ControlPanel } from "@/components/ControlPanel";
import { StatsCard } from "@/components/StatsCard";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Activity, Target, Zap, TrendingUp, Loader2, ShieldCheck, Sword, PenTool } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

type EncodedState = {
  grid_size: number;
  agents: string[];
  pos: Record<string, [number, number]>;
  victims: number[][];
  obstacles: number[][];
  t: number;
  victims_left: number;
};

type ResetResponse = { ok: boolean; error?: string; state: EncodedState };
type StepResponse = { ok: boolean; error?: string; state: EncodedState; rewards: Record<string, number>; done: boolean };
type ManualStepResponse = StepResponse;
type ToggleObstacleResponse = { ok: boolean; error?: string; state: EncodedState };

type OptionsState = {
  planner_assist: boolean;
  versus: boolean;
  manual_agent: number;
  record_demo: boolean;
};

type OptionsResponse = { ok: boolean; error?: string; options: OptionsState };

type SessionInfo = {
  initialized: boolean;
  greedy: boolean;
  n_agents: number | null;
  grid_size: number | null;
  agent_type: string | null;
};

type Availability = {
  checkpoints: Record<string, boolean>;
  logs: Record<string, boolean>;
};

type InfoResponse = { ok: boolean; session: SessionInfo; availability: Availability };

type MetricsSeries = {
  episode: number[];
  return: number[];
  loss: number[];
  victims_rescued: number[];
  victims_left: number[];
};

type MetricsSummary = {
  episodes: number;
  avg_return: number;
  success_rate: number;
  avg_rescued: number;
  path: string;
};

type MetricsResponse = { ok: boolean; error?: string; series: MetricsSeries; summary: MetricsSummary };
type EvaluationResponse = { ok: boolean; error?: string; summary: MetricsSummary };

const ACTION_KEY_MAP: Record<string, number> = {
  ArrowUp: 0,
  ArrowDown: 1,
  ArrowLeft: 2,
  ArrowRight: 3,
  w: 0,
  W: 0,
  s: 1,
  S: 1,
  a: 2,
  A: 2,
  d: 3,
  D: 3,
  " ": 4,
};

const toKey = (x: number, y: number) => `${x}-${y}`;
const fromKey = (key: string): [number, number] => {
  const [x, y] = key.split("-").map((v) => Number.parseInt(v, 10));
  return [x, y];
};

const movingAverage = (values: number[], windowSize: number) => {
  if (windowSize <= 1) {
    return [...values];
  }
  const result: number[] = [];
  let acc = 0;
  for (let i = 0; i < values.length; i += 1) {
    const val = Number.isFinite(values[i]) ? values[i] : 0;
    acc += val;
    if (i >= windowSize) {
      const prev = Number.isFinite(values[i - windowSize]) ? values[i - windowSize] : 0;
      acc -= prev;
    }
    const count = Math.min(windowSize, i + 1);
    result.push(acc / count);
  }
  return result;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: HeadersInit = { "Content-Type": "application/json", ...(init?.headers ?? {}) };
  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers, cache: "no-cache" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

const Index = () => {
  const [state, setState] = useState<EncodedState | null>(null);
  const [gridSize, setGridSize] = useState(10);
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(120);
  const [agents, setAgents] = useState(3);
  const [isGreedy, setIsGreedy] = useState(true);
  const [victimsLeft, setVictimsLeft] = useState(0);
  const [victimsRescued, setVictimsRescued] = useState(0);
  const [lastReward, setLastReward] = useState<number | null>(null);
  const [options, setOptions] = useState<OptionsState>({
    planner_assist: false,
    versus: false,
    manual_agent: 0,
    record_demo: false,
  });
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [availability, setAvailability] = useState<Availability | null>(null);
  const [manualMode, setManualMode] = useState(false);
  const [editObstacles, setEditObstacles] = useState(false);
  const [rescuedVictimPositions, setRescuedVictimPositions] = useState<number[][]>([]);

  const playLoopRef = useRef<number>();
  const manualInFlight = useRef(false);
  const allVictimsRef = useRef<Set<string>>(new Set());
  const rescuedVictimsRef = useRef<Set<string>>(new Set());

  const metricsQuery = useQuery({
    queryKey: ["metrics", agents],
    queryFn: async () => {
      const query = agents ? `?agents=${agents}` : "";
      const data = await api<MetricsResponse>(`/metrics${query}`);
      if (!data.ok) {
        throw new Error(data.error ?? "Metrics unavailable");
      }
      return data;
    },
    staleTime: 60_000,
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const evaluationQuery = useQuery({
    queryKey: ["evaluation", agents],
    queryFn: async () => {
      const query = agents ? `?agents=${agents}` : "";
      const data = await api<EvaluationResponse>(`/evaluation${query}`);
      if (!data.ok) {
        throw new Error(data.error ?? "Evaluation unavailable");
      }
      return data;
    },
    staleTime: 60_000,
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const syncState = useCallback((next: EncodedState) => {
    setState(next);
    setGridSize(next.grid_size);
    setStep(next.t);
    setVictimsLeft(next.victims_left);

    const victimKeys = new Set(next.victims.map(([x, y]) => toKey(x, y)));
    if (next.t === 0) {
      allVictimsRef.current = new Set(victimKeys);
      rescuedVictimsRef.current = new Set();
    } else {
      victimKeys.forEach((key) => allVictimsRef.current.add(key));
      allVictimsRef.current.forEach((key) => {
        if (!victimKeys.has(key)) {
          rescuedVictimsRef.current.add(key);
        }
      });
    }
    setRescuedVictimPositions(Array.from(rescuedVictimsRef.current).map(fromKey));
    setVictimsRescued(rescuedVictimsRef.current.size);
    setAgents(next.agents.length);
  }, []);

  const fetchInfo = useCallback(async () => {
    try {
      const data = await api<InfoResponse>("/info");
      if (data.ok) {
        setSessionInfo(data.session);
        setAvailability(data.availability);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load info";
      toast.error(message);
    }
  }, []);

  const fetchOptions = useCallback(async () => {
    try {
      const data = await api<OptionsResponse>("/options", { method: "POST", body: JSON.stringify({}) });
      if (data.ok) {
        setOptions(data.options);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load options";
      toast.error(message);
    }
  }, []);

  const handleReset = useCallback(
    async (override?: Partial<{ agents: number; greedy: boolean; seed: number | null }>) => {
      setIsPlaying(false);
      try {
        const payload = {
          agents: override?.agents ?? agents,
          greedy: override?.greedy ?? isGreedy,
          seed: override?.seed ?? undefined,
        };
        const data = await api<ResetResponse>("/reset", { method: "POST", body: JSON.stringify(payload) });
        if (!data.ok) {
          throw new Error(data.error ?? "Reset failed");
        }
        allVictimsRef.current = new Set(data.state.victims.map(([x, y]) => toKey(x, y)));
        rescuedVictimsRef.current = new Set();
        setRescuedVictimPositions([]);
        setLastReward(null);
        syncState(data.state);
        setAgents(payload.agents);
        setSessionInfo((prev) =>
          prev
            ? {
                ...prev,
                initialized: true,
                greedy: payload.greedy,
                n_agents: payload.agents,
                grid_size: data.state.grid_size,
              }
            : {
                initialized: true,
                greedy: payload.greedy,
                n_agents: payload.agents,
                grid_size: data.state.grid_size,
                agent_type: null,
              },
        );
        await fetchInfo();
        await metricsQuery.refetch();
        toast.success("Simulation reset");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to reset";
        toast.error(message);
      }
    },
    [agents, fetchInfo, isGreedy, metricsQuery, syncState],
  );

  const handleRandomize = useCallback(async () => {
    setIsPlaying(false);
    try {
      const payload = {
        agents,
        greedy: isGreedy,
      };
      const data = await api<ResetResponse>("/randomize", { method: "POST", body: JSON.stringify(payload) });
      if (!data.ok) {
        throw new Error(data.error ?? "Randomize failed");
      }
      allVictimsRef.current = new Set(data.state.victims.map(([x, y]) => toKey(x, y)));
      rescuedVictimsRef.current = new Set();
      setRescuedVictimPositions([]);
      setLastReward(null);
      syncState(data.state);
      toast.success("Environment randomized");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to randomize";
      toast.error(message);
    }
  }, [agents, isGreedy, syncState]);

  const updateOptions = useCallback(
    async (partial: Partial<OptionsState>) => {
      try {
        const data = await api<OptionsResponse>("/options", { method: "POST", body: JSON.stringify(partial) });
        if (!data.ok) {
          throw new Error(data.error ?? "Failed to update options");
        }
        setOptions(data.options);
        toast.success("Options updated");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to update options";
        toast.error(message);
      }
    },
    [],
  );

  const stepOnce = useCallback(async () => {
    try {
      const data = await api<StepResponse>("/step", { method: "POST", body: JSON.stringify({ greedy: isGreedy }) });
      if (!data.ok) {
        throw new Error(data.error ?? "Step failed");
      }
      syncState(data.state);
      const rewards = Object.values(data.rewards ?? {});
      if (rewards.length > 0) {
        const average = rewards.reduce((acc, val) => acc + val, 0) / rewards.length;
        setLastReward(average);
      }
      if (data.done) {
        setIsPlaying(false);
        toast.message("Episode finished", { description: "All victims rescued or episode truncated." });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Step failed";
      setIsPlaying(false);
      toast.error(message);
    }
  }, [isGreedy, syncState]);

  const manualStep = useCallback(
    async (action: number) => {
      if (manualInFlight.current) {
        return;
      }
      manualInFlight.current = true;
      try {
        const data = await api<ManualStepResponse>("/manual_step", {
          method: "POST",
          body: JSON.stringify({ action, greedy: isGreedy }),
        });
        if (!data.ok) {
          throw new Error(data.error ?? "Manual step failed");
        }
        syncState(data.state);
        const rewards = Object.values(data.rewards ?? {});
        if (rewards.length > 0) {
          const average = rewards.reduce((acc, val) => acc + val, 0) / rewards.length;
          setLastReward(average);
        }
        if (data.done) {
          setManualMode(false);
          toast.message("Episode finished");
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Manual step failed";
        toast.error(message);
      } finally {
        manualInFlight.current = false;
      }
    },
    [isGreedy, syncState],
  );

  const toggleObstacle = useCallback(
    async (x: number, y: number) => {
      try {
        const data = await api<ToggleObstacleResponse>("/toggle_obstacle", {
          method: "POST",
          body: JSON.stringify({ x, y }),
        });
        if (!data.ok) {
          throw new Error(data.error ?? "Toggle obstacle failed");
        }
        syncState(data.state);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Toggle obstacle failed";
        toast.error(message);
      }
    },
    [syncState],
  );

  useEffect(() => {
    void fetchInfo();
    void fetchOptions();
  }, [fetchInfo, fetchOptions]);

  useEffect(() => {
    if (!isPlaying) {
      if (playLoopRef.current) {
        window.clearTimeout(playLoopRef.current);
      }
      return;
    }
    let cancelled = false;
    const run = async () => {
      await stepOnce();
      if (!cancelled) {
        playLoopRef.current = window.setTimeout(run, speed);
      }
    };
    run();
    return () => {
      cancelled = true;
      if (playLoopRef.current) {
        window.clearTimeout(playLoopRef.current);
      }
    };
  }, [isPlaying, speed, stepOnce]);

  useEffect(() => {
    if (manualMode && isPlaying) {
      setIsPlaying(false);
    }
  }, [isPlaying, manualMode]);

  useEffect(() => {
    if (!manualMode) {
      return;
    }
    const handleKey = (event: KeyboardEvent) => {
      const action = ACTION_KEY_MAP[event.key];
      if (action === undefined) {
        return;
      }
      event.preventDefault();
      void manualStep(action);
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
    };
  }, [manualMode, manualStep]);

  const gridEntities = useMemo(() => {
    if (!state) {
      return [];
    }
    const drones = Object.values(state.pos ?? {}).map(([x, y]) => ({ type: "drone" as const, x, y }));
    const victimsRemaining = (state.victims ?? []).map(([x, y]) => ({ type: "victim-left" as const, x, y }));
    const rescued = rescuedVictimPositions.map(([x, y]) => ({ type: "victim-rescued" as const, x, y }));
    return [...drones, ...victimsRemaining, ...rescued];
  }, [rescuedVictimPositions, state]);

  const obstacles = useMemo(() => state?.obstacles ?? [], [state]);

  const metricsData = metricsQuery.data;
  const metricsSummary = evaluationQuery.data?.summary;

  const buildChartData = useCallback((episodes: number[], values: number[], windowSize: number) => {
    const smooth = movingAverage(values, windowSize);
    return episodes.map((episode, idx) => ({
      episode,
      value: values[idx],
      smoothed: smooth[idx],
    }));
  }, []);

  const returnChart = useMemo(() => {
    if (!metricsData) return [];
    return buildChartData(metricsData.series.episode, metricsData.series.return, 25);
  }, [buildChartData, metricsData]);

  const lossChart = useMemo(() => {
    if (!metricsData) return [];
    return buildChartData(metricsData.series.episode, metricsData.series.loss, 25);
  }, [buildChartData, metricsData]);

  const rescuedChart = useMemo(() => {
    if (!metricsData) return [];
    return buildChartData(metricsData.series.episode, metricsData.series.victims_rescued, 15);
  }, [buildChartData, metricsData]);

  const leftChart = useMemo(() => {
    if (!metricsData) return [];
    return buildChartData(metricsData.series.episode, metricsData.series.victims_left, 15);
  }, [buildChartData, metricsData]);

  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    const start = async () => {
      if (!state) {
        await handleReset();
      }
      setManualMode(false);
      setIsPlaying(true);
    };
    void start();
  }, [handleReset, isPlaying, state]);

  const handleAgentsChange = useCallback(
    (value: string) => {
      const parsed = Number.parseInt(value, 10);
      if (Number.isNaN(parsed)) {
        return;
      }
      setAgents(parsed);
      void handleReset({ agents: parsed });
    },
    [handleReset],
  );

  const handleGreedyChange = useCallback((value: boolean) => {
    setIsGreedy(value);
    setSessionInfo((prev) => (prev ? { ...prev, greedy: value } : prev));
  }, []);

  const handleCellClick = useCallback(
    (x: number, y: number) => {
      if (!editObstacles) {
        return;
      }
      void toggleObstacle(x, y);
    },
    [editObstacles, toggleObstacle],
  );

  const manualAgentOptions = useMemo(() => {
    const count = state?.agents.length ?? agents;
    return Array.from({ length: count }, (_, idx) => idx);
  }, [agents, state]);

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="mb-2 text-5xl font-bold bg-gradient-to-r from-primary via-chart-purple to-accent bg-clip-text text-transparent">
              Multi-Drone Rescue
            </h1>
            <p className="text-lg text-muted-foreground">
              Real-time coordination dashboard for DRQN-QMIX agents in a search-and-rescue grid.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {metricsSummary ? (
              <>
                <Badge variant="secondary" className="px-4 py-2 backdrop-blur-sm">
                  Episodes: {metricsSummary.episodes}
                </Badge>
                <Badge variant="secondary" className="px-4 py-2 backdrop-blur-sm">
                  Avg Return: {metricsSummary.avg_return.toFixed(1)}
                </Badge>
                <Badge variant="default" className="px-4 py-2 bg-gradient-to-r from-primary to-primary/80 shadow-[0_0_20px_rgba(96,165,250,0.3)]">
                  Success: {(metricsSummary.success_rate * 100).toFixed(1)}%
                </Badge>
              </>
            ) : (
              <Badge variant="secondary" className="px-4 py-2 backdrop-blur-sm">
                Loading metrics...
              </Badge>
            )}
          </div>
        </div>

        <Tabs defaultValue="live" className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="live">Live</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="overview">Overview</TabsTrigger>
          </TabsList>

          <TabsContent value="live" className="space-y-6">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
              <StatsCard title="Current Step" value={step} icon={Activity} />
              <StatsCard title="Victims Left" value={victimsLeft} icon={Target} />
              <StatsCard title="Victims Rescued" value={victimsRescued} icon={Zap} />
              <StatsCard
                title="Last Reward (avg)"
                value={lastReward !== null ? lastReward.toFixed(2) : "—"}
                icon={TrendingUp}
              />
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2 space-y-4">
                <DroneGrid
                  gridSize={gridSize}
                  entities={gridEntities}
                  obstacles={obstacles}
                  editable={editObstacles}
                  onCellClick={handleCellClick}
                />
                {editObstacles ? (
                  <p className="text-xs text-muted-foreground text-center">
                    Click cells to toggle obstacles. Agents will avoid obstacles and victims cannot be covered.
                  </p>
                ) : null}
              </div>
              <div className="flex flex-col gap-4">
                <ControlPanel
                  isPlaying={isPlaying}
                  onPlayPause={handlePlayPause}
                  onReset={() => {
                    void handleReset();
                  }}
                  speed={speed}
                  onSpeedChange={(v) => setSpeed(v[0])}
                  agents={agents}
                  onAgentsChange={handleAgentsChange}
                  isGreedy={isGreedy}
                  onGreedyChange={handleGreedyChange}
                />

                <Card className="shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50">
                  <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle className="text-sm bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
                      Session Info
                    </CardTitle>
                    <Button size="sm" variant="outline" onClick={() => void handleRandomize()}>
                      Randomize
                    </Button>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Initialized</span>
                      <Badge variant={sessionInfo?.initialized ? "default" : "outline"}>
                        {sessionInfo?.initialized ? "Yes" : "No"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Agents</span>
                      <span className="font-medium">{state?.agents.length ?? agents}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Grid Size</span>
                      <span className="font-medium">{state?.grid_size ?? sessionInfo?.grid_size ?? "—"}×{state?.grid_size ?? sessionInfo?.grid_size ?? "—"}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Algorithm</span>
                      <span className="font-medium">{sessionInfo?.agent_type ?? "DRQN-QMIX"}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Epsilon</span>
                      <span className="font-medium">{isGreedy ? "0.0" : "0.1"}</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50">
                  <CardHeader>
                    <CardTitle className="text-sm bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
                      Advanced Controls
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4 text-sm">
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <ShieldCheck className="h-4 w-4 text-primary" />
                        <span>Planner Assist</span>
                      </div>
                      <Switch
                        checked={options.planner_assist}
                        onCheckedChange={(checked) => void updateOptions({ planner_assist: checked })}
                      />
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Sword className="h-4 w-4 text-primary" />
                        <span>Versus Mode</span>
                      </div>
                      <Switch
                        checked={options.versus}
                        onCheckedChange={(checked) => void updateOptions({ versus: checked })}
                      />
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-muted-foreground">Record Demo</span>
                      <Switch
                        checked={options.record_demo}
                        onCheckedChange={(checked) => void updateOptions({ record_demo: checked })}
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-muted-foreground">Manual Agent</span>
                      <Select
                        value={String(options.manual_agent ?? 0)}
                        onValueChange={(value) => void updateOptions({ manual_agent: Number.parseInt(value, 10) })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select agent" />
                        </SelectTrigger>
                        <SelectContent>
                          {manualAgentOptions.map((idx) => (
                            <SelectItem key={idx} value={String(idx)}>
                              Agent {idx + 1}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-muted-foreground">Manual Mode</span>
                      <Switch
                        checked={manualMode}
                        onCheckedChange={(checked) => {
                          if (checked && !state) {
                            toast.info("Reset the simulation before entering manual mode");
                            void handleReset();
                            return;
                          }
                          setManualMode(checked);
                        }}
                      />
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <PenTool className="h-4 w-4 text-primary" />
                        <span>Edit Obstacles</span>
                      </div>
                      <Switch checked={editObstacles} onCheckedChange={setEditObstacles} />
                    </div>
                    {manualMode ? (
                      <p className="text-xs text-muted-foreground">
                        Use arrow keys or WASD to move the selected agent. Spacebar makes the agent stay in place.
                      </p>
                    ) : null}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="metrics" className="space-y-6">
            {metricsQuery.isLoading ? (
              <Card className="flex h-48 items-center justify-center shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
              </Card>
            ) : metricsQuery.isError || !metricsData ? (
              <Card className="shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50">
                <CardContent className="py-16 text-center text-muted-foreground">
                  Failed to load metrics. Ensure training logs exist under <code className="rounded bg-muted px-1 py-0.5">logs/</code>.
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <MetricsChart
                  title="Return per Episode"
                  data={returnChart}
                  yAxisLabel="Return"
                  series={[
                    {
                      dataKey: "value",
                      name: "Return",
                      color: "hsl(var(--chart-green))",
                      area: true,
                    },
                    {
                      dataKey: "smoothed",
                      name: "Moving Avg (25)",
                      color: "hsl(var(--chart-blue))",
                      strokeDasharray: "6 6",
                    },
                  ]}
                />
                <MetricsChart
                  title="Loss per Episode"
                  data={lossChart}
                  yAxisLabel="Loss"
                  series={[
                    {
                      dataKey: "value",
                      name: "Loss",
                      color: "hsl(var(--chart-orange))",
                      area: true,
                    },
                    {
                      dataKey: "smoothed",
                      name: "Moving Avg (25)",
                      color: "hsl(var(--chart-blue))",
                      strokeDasharray: "6 6",
                    },
                  ]}
                />
                <MetricsChart
                  title="Victims Rescued"
                  data={rescuedChart}
                  yAxisLabel="Count"
                  series={[
                    {
                      dataKey: "value",
                      name: "Per Episode",
                      color: "hsl(var(--chart-blue))",
                      area: true,
                    },
                    {
                      dataKey: "smoothed",
                      name: "Moving Avg (15)",
                      color: "hsl(var(--chart-green))",
                      strokeDasharray: "6 6",
                    },
                  ]}
                />
                <MetricsChart
                  title="Victims Left"
                  data={leftChart}
                  yAxisLabel="Count"
                  series={[
                    {
                      dataKey: "value",
                      name: "Per Episode",
                      color: "hsl(var(--chart-red))",
                      area: true,
                    },
                    {
                      dataKey: "smoothed",
                      name: "Moving Avg (15)",
                      color: "hsl(var(--chart-orange))",
                      strokeDasharray: "6 6",
                    },
                  ]}
                />
              </div>
            )}
          </TabsContent>

          <TabsContent value="overview" className="space-y-6">
            <Card className="shadow-[var(--shadow-card)] bg-card/50 backdrop-blur-sm border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  <span className="bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
                    Project Overview
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="mb-2 font-semibold">Description</h3>
                  <p className="leading-relaxed text-muted-foreground">
                    Multi-agent search-and-rescue with DRQN-QMIX. Drones operate under partial observability with shaped
                    rewards, action masking, and centralized training via a mixing network. Execution is decentralized
                    and can be augmented with planner assist, human-in-the-loop control, and obstacle editing.
                  </p>
                </div>

                <div>
                  <h3 className="mb-3 font-semibold">Current Session</h3>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Initialized:</span>
                        <Badge variant={sessionInfo?.initialized ? "default" : "outline"}>
                          {sessionInfo?.initialized ? "Yes" : "No"}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Agents:</span>
                        <span className="font-medium">{sessionInfo?.n_agents ?? agents}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Grid Size:</span>
                        <span className="font-medium">{sessionInfo?.grid_size ?? state?.grid_size ?? "—"}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Agent Type:</span>
                        <span className="font-medium">{sessionInfo?.agent_type ?? "DRQN"}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Greedy:</span>
                        <Badge variant="secondary">{(sessionInfo?.greedy ?? isGreedy) ? "true" : "false"}</Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Last Reward Avg:</span>
                        <span className="font-medium">{lastReward !== null ? lastReward.toFixed(2) : "—"}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="mb-3 font-semibold">Artifacts</h3>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">Checkpoints:</span>
                      <div className="flex gap-2">
                        {availability
                          ? Object.entries(availability.checkpoints).map(([label, value]) => (
                              <Badge key={label} variant={value ? "secondary" : "outline"}>
                                {label.replace("_", " ")} {value ? "✓" : "×"}
                              </Badge>
                            ))
                          : <span className="text-muted-foreground">—</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">Logs:</span>
                      <div className="flex gap-2">
                        {availability
                          ? Object.entries(availability.logs).map(([label, value]) => (
                              <Badge key={label} variant={value ? "secondary" : "outline"}>
                                {label.replace("_", " ")} {value ? "✓" : "×"}
                              </Badge>
                            ))
                          : <span className="text-muted-foreground">—</span>}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="border-t pt-4">
                  <p className="text-sm text-muted-foreground italic">
                    Use the Live tab to step through episodes, fine-tune planner assist, and intervene manually.
                    Metrics summarise historical training performance sourced from the logs directory.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;

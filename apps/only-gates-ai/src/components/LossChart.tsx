import { createMemo, For, Show } from "solid-js";

import type { TrainingPoint } from "../lib/trainTinyGpt";

interface LossChartProps {
  readonly points: readonly TrainingPoint[];
  readonly bestStep?: number | null;
}

const WIDTH = 720;
const HEIGHT = 210;
const PADDING = { left: 48, right: 14, top: 14, bottom: 28 };

function downsample(
  points: readonly TrainingPoint[],
  limit: number,
): readonly TrainingPoint[] {
  if (points.length <= limit) {
    return points;
  }
  const stride = Math.ceil(points.length / limit);
  const sampled: TrainingPoint[] = [];
  for (let index = 0; index < points.length; index += stride) {
    sampled.push(points[index]!);
  }
  const last = points[points.length - 1]!;
  if (sampled[sampled.length - 1] !== last) {
    sampled.push(last);
  }
  return sampled;
}

function linePath(
  points: readonly { readonly step: number; readonly loss: number }[],
  x: (step: number) => number,
  y: (loss: number) => number,
): string {
  return points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"}${x(point.step).toFixed(2)},${y(
          point.loss,
        ).toFixed(2)}`,
    )
    .join(" ");
}

export default function LossChart(props: LossChartProps) {
  const view = createMemo(() => {
    const points = props.points;
    if (points.length === 0) {
      return null;
    }

    const lastStep = points[points.length - 1]!.step;
    const stepMax = Math.max(1, lastStep);
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    const include = (value: number | undefined): void => {
      if (value === undefined || !Number.isFinite(value)) return;
      if (value < min) min = value;
      if (value > max) max = value;
    };
    for (const point of points) {
      include(point.loss);
      include(point.smoothedLoss);
      include(point.valLoss);
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return null;
    }
    if (max - min < 1e-6) {
      max = min + 1;
    }
    const margin = (max - min) * 0.08;
    min = Math.max(0, min - margin);
    max += margin;

    const plotWidth = WIDTH - PADDING.left - PADDING.right;
    const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;
    const x = (step: number): number =>
      PADDING.left + (step / stepMax) * plotWidth;
    const y = (loss: number): number =>
      PADDING.top + (1 - (loss - min) / (max - min)) * plotHeight;

    const sampled = downsample(points, 420);
    const raw = linePath(
      sampled
        .filter((point) => Number.isFinite(point.loss))
        .map((point) => ({ step: point.step, loss: point.loss })),
      x,
      y,
    );
    const smooth = linePath(
      sampled
        .filter((point) => Number.isFinite(point.smoothedLoss))
        .map((point) => ({
          step: point.step,
          loss: point.smoothedLoss,
        })),
      x,
      y,
    );
    const validation = points
      .filter(
        (point): point is TrainingPoint & { readonly valLoss: number } =>
          point.valLoss !== undefined && Number.isFinite(point.valLoss),
      )
      .map((point) => ({ step: point.step, loss: point.valLoss }));
    const validationPath = linePath(validation, x, y);

    return {
      min,
      max,
      stepMax,
      x,
      y,
      raw,
      smooth,
      validation,
      validationPath,
      bestX:
        props.bestStep !== null &&
        props.bestStep !== undefined &&
        props.bestStep > 0
          ? x(props.bestStep)
          : null,
    };
  });

  return (
    <div class="loss-chart">
      <Show
        when={view()}
        fallback={
          <div class="loss-chart-empty">
            Loss will appear here as soon as training starts.
          </div>
        }
      >
        {(chart) => (
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            role="img"
            aria-label={`Training loss over ${props.points.length} recorded steps`}
          >
            <title>Training loss</title>
            <desc>
              Raw training loss, smoothed training loss, and validation loss
              over optimizer steps. The dashed vertical line marks the best
              validation loss.
            </desc>

            <For each={[0, 0.25, 0.5, 0.75, 1]}>
              {(fraction) => {
                const lineY =
                  PADDING.top +
                  fraction * (HEIGHT - PADDING.top - PADDING.bottom);
                return (
                  <line
                    x1={PADDING.left}
                    x2={WIDTH - PADDING.right}
                    y1={lineY}
                    y2={lineY}
                    class="loss-grid"
                  />
                );
              }}
            </For>

            <text
              x={PADDING.left - 8}
              y={PADDING.top + 4}
              text-anchor="end"
              class="loss-label"
            >
              {chart().max.toFixed(2)}
            </text>
            <text
              x={PADDING.left - 8}
              y={HEIGHT - PADDING.bottom}
              text-anchor="end"
              class="loss-label"
            >
              {chart().min.toFixed(2)}
            </text>
            <text
              x={PADDING.left}
              y={HEIGHT - 8}
              class="loss-label"
            >
              1
            </text>
            <text
              x={WIDTH - PADDING.right}
              y={HEIGHT - 8}
              text-anchor="end"
              class="loss-label"
            >
              {chart().stepMax}
            </text>

            <path d={chart().raw} class="loss-raw" />
            <path d={chart().smooth} class="loss-smooth" />
            <Show when={chart().validation.length > 0}>
              <path d={chart().validationPath} class="loss-val" />
              <For each={chart().validation}>
                {(point) => (
                  <circle
                    cx={chart().x(point.step)}
                    cy={chart().y(point.loss)}
                    r={2.5}
                    class="loss-val-dot"
                  />
                )}
              </For>
            </Show>
            <Show when={chart().bestX !== null}>
              <line
                x1={chart().bestX!}
                x2={chart().bestX!}
                y1={PADDING.top}
                y2={HEIGHT - PADDING.bottom}
                class="loss-best"
              />
            </Show>
          </svg>
        )}
      </Show>
      <div class="loss-legend" aria-hidden="true">
        <span class="loss-legend-raw">raw</span>
        <span class="loss-legend-smooth">smoothed</span>
        <span class="loss-legend-val">validation</span>
      </div>
    </div>
  );
}

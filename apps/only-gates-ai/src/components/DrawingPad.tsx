import { createEffect, Show } from "solid-js";

import type { Point, Strokes } from "../lib/types";
import GridPreview from "./GridPreview";

const DRAW_SIZE = 360;
const BACKING_SCALE = 2;
const BACKING_SIZE = DRAW_SIZE * BACKING_SCALE;

interface DrawingPadProps {
  readonly strokes: Strokes;
  readonly input: readonly boolean[];
  readonly gridSize: number;
  readonly color?: string;
  readonly disabled?: boolean;
  readonly onStrokeStart: (point: Point) => void;
  readonly onStrokeMove: (point: Point) => void;
  readonly onStrokeEnd: () => void;
  readonly onUndo: () => void;
  readonly onClear: () => void;
}

function canvasPoint(
  event: PointerEvent & { currentTarget: HTMLCanvasElement },
): Point {
  const bounds = event.currentTarget.getBoundingClientRect();
  return {
    x: ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * DRAW_SIZE,
    y: ((event.clientY - bounds.top) / Math.max(1, bounds.height)) * DRAW_SIZE,
  };
}

export default function DrawingPad(props: DrawingPadProps) {
  let canvas: HTMLCanvasElement | undefined;
  let drawing = false;

  createEffect(
    () => props.strokes,
    (strokes) => {
      const element = canvas;
      if (!element) {
        return;
      }
      const context = element.getContext("2d");
      if (!context) {
        return;
      }

      context.setTransform(BACKING_SCALE, 0, 0, BACKING_SCALE, 0, 0);
      context.fillStyle = "#0a0e13";
      context.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);

      // A whisper of a guide, useful for scale but not distracting.
      context.strokeStyle = "rgba(255, 255, 255, 0.06)";
      context.lineWidth = 1;
      for (let index = 1; index < 4; index++) {
        const offset = (DRAW_SIZE / 4) * index;
        context.beginPath();
        context.moveTo(offset, 0);
        context.lineTo(offset, DRAW_SIZE);
        context.moveTo(0, offset);
        context.lineTo(DRAW_SIZE, offset);
        context.stroke();
      }

      context.strokeStyle = "#f2f5f8";
      context.fillStyle = "#f2f5f8";
      context.lineWidth = 5.5;
      context.lineCap = "round";
      context.lineJoin = "round";

      for (const stroke of strokes) {
        if (stroke.length === 0) {
          continue;
        }
        if (stroke.length === 1) {
          const point = stroke[0]!;
          context.beginPath();
          context.arc(point.x, point.y, 2.75, 0, Math.PI * 2);
          context.fill();
          continue;
        }
        context.beginPath();
        const first = stroke[0]!;
        context.moveTo(first.x, first.y);
        for (let index = 1; index < stroke.length; index++) {
          const point = stroke[index]!;
          context.lineTo(point.x, point.y);
        }
        context.stroke();
      }
    },
  );

  function handlePointerDown(
    event: PointerEvent & { currentTarget: HTMLCanvasElement },
  ): void {
    if (props.disabled) {
      return;
    }
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }
    event.preventDefault();
    drawing = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    props.onStrokeStart(canvasPoint(event));
  }

  function handlePointerMove(
    event: PointerEvent & { currentTarget: HTMLCanvasElement },
  ): void {
    if (!drawing || props.disabled) {
      return;
    }
    event.preventDefault();
    props.onStrokeMove(canvasPoint(event));
  }

  function finishStroke(
    event: PointerEvent & { currentTarget: HTMLCanvasElement },
  ): void {
    if (!drawing) {
      return;
    }
    drawing = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    props.onStrokeEnd();
  }

  return (
    <>
      <div class="draw-surface">
        <div class="canvas-wrap">
          <canvas
            ref={(element) => {
              canvas = element;
            }}
            class="drawing-canvas"
            width={BACKING_SIZE}
            height={BACKING_SIZE}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={finishStroke}
            onPointerCancel={finishStroke}
            onLostPointerCapture={() => {
              drawing = false;
            }}
            aria-label="Drawing canvas"
          />
          <Show when={props.strokes.length === 0}>
            <div class="drawing-placeholder">
              <strong>Draw a symbol</strong>
              <span>Use one or several strokes</span>
            </div>
          </Show>
        </div>

        <aside class="preview-panel">
          <span class="preview-label">Circuit input</span>
          <GridPreview
            input={props.input}
            gridSize={props.gridSize}
            color={props.color}
            size={150}
            showGrid={props.gridSize <= 32}
          />
          <span class="preview-meta">
            {props.gridSize} × {props.gridSize}
          </span>
          <span class="preview-meta">
            {props.input.filter(Boolean).length} / {props.gridSize * props.gridSize} cells
          </span>
        </aside>
      </div>

      <div class="draw-actions">
        <button
          type="button"
          class="button ghost"
          onClick={props.onUndo}
          disabled={props.disabled || props.strokes.length === 0}
        >
          Undo stroke
        </button>
        <button
          type="button"
          class="button ghost"
          onClick={props.onClear}
          disabled={props.disabled || props.strokes.length === 0}
        >
          Clear canvas
        </button>
      </div>
    </>
  );
}

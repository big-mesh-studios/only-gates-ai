import { createEffect } from "solid-js";

interface GridPreviewProps {
  readonly input: readonly boolean[];
  readonly gridSize: number;
  readonly color?: string;
  readonly size?: number;
  readonly showGrid?: boolean;
  readonly class?: string;
}

/**
 * Draws the boolean input grid exactly as the circuit sees it. The same
 * component is used for the live preview and for saved-drawing thumbnails.
 */
export default function GridPreview(props: GridPreviewProps) {
  let canvas: HTMLCanvasElement | undefined;

  createEffect(
    () => ({
      input: props.input,
      gridSize: props.gridSize,
      color: props.color,
      size: props.size ?? 160,
      showGrid: props.showGrid ?? false,
    }),
    ({ input, gridSize, color, size, showGrid }) => {
      const element = canvas;
      if (!element) {
        return;
      }

      if (element.width !== size) {
        element.width = size;
      }
      if (element.height !== size) {
        element.height = size;
      }

      const context = element.getContext("2d");
      if (!context) {
        return;
      }

      context.clearRect(0, 0, size, size);
      context.fillStyle = "#0a0e13";
      context.fillRect(0, 0, size, size);

      const cellSize = size / gridSize;
      context.fillStyle = color ?? "#7ee787";
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          if (input[y * gridSize + x]) {
            context.fillRect(
              x * cellSize,
              y * cellSize,
              Math.ceil(cellSize + 0.5),
              Math.ceil(cellSize + 0.5),
            );
          }
        }
      }

      if (showGrid && cellSize >= 3) {
        context.strokeStyle = "rgba(255, 255, 255, 0.055)";
        context.lineWidth = 1;
        context.beginPath();
        for (let index = 0; index <= gridSize; index++) {
          const offset = Math.round(index * cellSize) + 0.5;
          context.moveTo(offset, 0);
          context.lineTo(offset, size);
          context.moveTo(0, offset);
          context.lineTo(size, offset);
        }
        context.stroke();
      }
    },
  );

  return (
    <canvas
      ref={(element) => {
        canvas = element;
      }}
      class={props.class ? ["grid-preview", props.class] : "grid-preview"}
      style={{
        width: "100%",
        "max-width": `${props.size ?? 160}px`,
        height: "auto",
        "aspect-ratio": "1",
      }}
      role="img"
      aria-label="Downsampled drawing input grid"
    />
  );
}

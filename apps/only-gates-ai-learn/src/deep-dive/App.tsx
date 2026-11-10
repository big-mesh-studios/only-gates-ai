import { For } from "solid-js";

import { NAV } from "./content";
import { Contract } from "./chapters/contract";
import { ModelChapter, FloatChapter } from "./chapters/model";
import {
  FixedChapter,
  QuantizeChapter,
  MatmulChapter,
  NormChapter,
  GeluChapter,
} from "./chapters/arithmetic";
import {
  AttentionChapter,
  ForwardChapter,
  SamplingChapter,
} from "./chapters/transformer";
import {
  GenerationChapter,
  TrainingChapter,
  VerifyChapter,
  BeyondChapter,
} from "./chapters/end-to-end";
import { Appendix } from "./chapters/appendix";

const isDev = import.meta.env.DEV;
const demoHref = isDev ? "http://localhost:5174/" : "../../";
// Both in dev (/deep-dive/) and in the assembled site (/learn/deep-dive/),
// the field notes live one directory up.
const notesHref = "../";
const repo = "https://github.com/clinuxrulz/only-gates-ai";

function Header() {
  return (
    <header class="masthead">
      <a class="brand" href="#top" aria-label="Integer GPT deep dive, back to top">
        <span class="brand-mark" aria-hidden="true">⌁</span>
        <span>only-gates-ai</span>
      </a>
      <nav aria-label="Page links">
        <a href={notesHref}>Field notes</a>
        <a href={`${repo}/tree/main/src`}>Source</a>
        <a href={`${repo}/tree/main/examples/integer-gpt`}>Example</a>
      </nav>
      <a class="nav-cta" href={demoHref}>Open the demo</a>
    </header>
  );
}

function Hero() {
  return (
    <section class="deep-hero" id="top">
      <p class="eyebrow">Integer GPT · build it from scratch</p>
      <h1>
        A GPT that does arithmetic
        <span class="accent"> with integers only.</span>
      </h1>
      <p class="lede">
        This is the long version. It starts with the float checkpoint and ends
        with an integer-only forward pass you can rebuild from the listings on
        this page: int8 weights, Q16 activations, int32 accumulation, integer
        RMSNorm, a GELU table, DI-Exp softmax, and deterministic sampling.
      </p>
      <div class="actions">
        <a class="button" href="#contract">Start reading</a>
        <a class="button ghost" href="#appendix">Jump to the full source</a>
        <a class="button ghost" href={demoHref}>Try the demo</a>
      </div>
      <ul class="chips" aria-label="What this page covers">
        <li class="chip ready">Integer inference · working</li>
        <li class="chip prototype">Full source · on this page</li>
        <li class="chip roadmap">Runnable C example</li>
      </ul>
    </section>
  );
}

function Sidebar() {
  return (
    <aside class="deep-sidebar" aria-label="Chapter contents">
      <p class="toc-title">Contents</p>
      <nav>
        <For each={NAV}>
          {(item) => (
            <a href={item.href}>
              <span class="toc-number">{item.number}</span>
              <span>{item.label}</span>
            </a>
          )}
        </For>
      </nav>
      <div class="toc-note">
        <p>
          The page reads top to bottom. Each chapter ends with a test you can
          run and the exact production function it explains.
        </p>
        <p>
          <a href={notesHref}>Read the shorter field notes →</a>
        </p>
      </div>
    </aside>
  );
}

function Footer() {
  return (
    <footer class="footer">
      <p>
        <strong>only-gates-ai</strong> — integer inference today, gate lowering
        next. The engine source on this page is the shipping C/WASM path. The
        TypeScript float path exists for comparison, not as the integer
        reference.
      </p>
      <p>
        <a href={repo}>Source</a>
        <span class="dot">·</span>
        <a href={`${repo}/tree/main/src`}>Engine</a>
        <span class="dot">·</span>
        <a href={`${repo}/tree/main/examples/integer-gpt`}>Example</a>
        <span class="dot">·</span>
        <a href={notesHref}>Field notes</a>
        <span class="dot">·</span>
        <a href="#top">Back to top</a>
      </p>
    </footer>
  );
}

export function DeepDiveApp() {
  return (
    <>
      <a class="skip-link" href="#main">
        Skip to the tutorial
      </a>
      <Header />
      <Hero />
      <div class="deep-layout">
        <Sidebar />
        <main class="deep-article" id="main">
          <Contract />
          <ModelChapter />
          <FloatChapter />
          <FixedChapter />
          <QuantizeChapter />
          <MatmulChapter />
          <NormChapter />
          <GeluChapter />
          <AttentionChapter />
          <ForwardChapter />
          <SamplingChapter />
          <GenerationChapter />
          <TrainingChapter />
          <VerifyChapter />
          <BeyondChapter />
          <Appendix />
        </main>
      </div>
      <Footer />
    </>
  );
}

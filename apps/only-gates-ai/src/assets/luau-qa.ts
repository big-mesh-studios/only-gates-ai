import type { QaDocument } from "../lib/qaRuntime.ts";

/*
 * A curated Luau Q&A corpus.
 *
 * Answers are written for this demo and checked against the official Luau
 * documentation. Each document links back to the relevant page. Questions are
 * deliberately varied so retrieval can recognise the same concept from a
 * syntax question, a type-system question, or a first-time-user phrasing.
 *
 * Sources (pinned to the official site):
 *   https://luau.org/
 */

const SITE = "https://luau.org";
const SYNTAX = `${SITE}/syntax`;
const TYPES = `${SITE}/types`;
const BASIC_TYPES = `${SITE}/types/basic-types`;
const TABLE_TYPES = `${SITE}/types/tables`;
const UNIONS = `${SITE}/types/unions-and-intersections`;
const GENERICS = `${SITE}/types/generics`;
const REFINEMENTS = `${SITE}/types/type-refinements`;
const OOP = `${SITE}/types/object-oriented-programs`;
const TYPE_FUNCTIONS = `${SITE}/types/type-functions`;
const TYPES_LIBRARY = `${SITE}/types-library`;
const CONSIDERATIONS = `${SITE}/types/considerations`;
const LIBRARY = `${SITE}/library`;
const LINT = `${SITE}/lint`;
const COMPATIBILITY = `${SITE}/compatibility`;
const PERFORMANCE = `${SITE}/performance`;
const ATTRIBUTES = `${SITE}/attributes`;
const GETTING_STARTED = `${SITE}/getting-started`;
const SANDBOX = `${SITE}/sandbox`;

export const LUAU_QA_DOCUMENTS: readonly QaDocument[] = [
  {
    id: "luau-overview",
    questions: [
      "What is Luau?",
      "What is the Luau programming language?",
      "Is Luau a version of Lua?",
      "What is the difference between Lua and Luau?",
      "Why does Luau exist?",
      "What scripting language does Roblox use?",
      "Is Luau Lua 5.1?",
      "What features does Luau add to Lua?",
    ],
    answer: `Luau is a fast, small, safe, gradually typed embeddable scripting language derived from Lua 5.1. It keeps Lua's syntax and semantics where it makes sense, removes features that are unsafe for sandboxing, and adds a gradual type system, a faster bytecode VM and garbage collector, and extra language features.

Those additions include \`continue\`, compound assignments, \`const\` bindings, if-then-else expressions, generalized iteration, string interpolation, floor division, attributes, the \`vector\` and \`buffer\` types, and a richer standard library. Roblox uses Luau, but Luau is also a standalone language with its own \`luau\` command line tool.

Luau is backwards-compatible with Lua 5.1 where possible, but it is not a full superset of later Lua versions. Features such as 64-bit integers, bitwise operators, and \`goto\` are intentionally absent.`,
    source: `${GETTING_STARTED}#language-overview`,
    tags: ["overview", "language"],
    keywords: ["Luau", "Lua", "Roblox", "gradual typing", "embedded"],
  },
  {
    id: "running-luau",
    questions: [
      "How do I run a Luau script?",
      "How do I install Luau?",
      "What is luau-analyze?",
      "How do I type check a Luau file?",
      "What file extension do Luau scripts use?",
      "How do I run Luau from the command line?",
      "Where do I download the Luau CLI?",
      "How do I test a Luau snippet online?",
    ],
    answer: `Download the \`luau\` and \`luau-analyze\` binaries from a Luau GitHub release. Luau scripts conventionally use the \`.luau\` extension.

\`\`\`sh
luau test.luau
luau-analyze test.luau
\`\`\`

\`luau\` runs the script. \`luau-analyze\` runs the type checker and linter. You can also try snippets in the official playground at https://play.luau.org. Roblox Studio integrates Luau directly, so no separate install is needed there.`,
    source: `${GETTING_STARTED}#language-overview`,
    tags: ["tooling", "cli"],
    keywords: ["luau CLI", "luau-analyze", "playground", "install"],
  },
  {
    id: "type-check-directives",
    questions: [
      "What are --!strict and --!nonstrict?",
      "How do I enable strict mode in Luau?",
      "What is --!nocheck?",
      "What is the default type checking mode in Luau?",
      "How do I disable type checking for a file?",
      "Where do I put --!strict?",
      "What does nonstrict mode do?",
      "What is the difference between strict and nonstrict?",
    ],
    answer: `Luau has three file-level type checking modes. Put the directive at the top of the file:

- \`--!nocheck\` turns the type checker off for that file.
- \`--!nonstrict\` is the default. It reports errors only when it can prove the program will fail at runtime, and falls back to \`any\` when it cannot infer a type.
- \`--!strict\` tracks types across statements and reports errors when a program might fail, even if a particular run would not.

In strict mode, assigning a number to a variable and later adding a string to it is an error. In nonstrict mode the variable may become \`any\`, so the same code can pass analysis and still fail at runtime. The type checker is separate from the linter: \`--!nocheck\` does not disable lint warnings, and \`--!nolint\` does not disable type checking.`,
    source: `${TYPES}#type-inference-modes`,
    tags: ["types", "strict", "tooling"],
    keywords: ["--!strict", "--!nonstrict", "--!nocheck", "type checking mode"],
  },
  {
    id: "string-escapes",
    questions: [
      "How do I write a Unicode escape in Luau?",
      "Does Luau support hexadecimal escapes in strings?",
      "What does \\z do in a Luau string?",
      "How do I put a Unicode character in a Luau string?",
      "What string escape sequences does Luau support?",
      "How do I break a long string across lines in Luau?",
      "Does Luau support \\u escapes?",
    ],
    answer: `Luau supports the Lua 5.3 string escape extensions in addition to the usual Lua escapes:

- \`\\xAB\` inserts the byte with code 0xAB.
- \`\\u{ABC}\` inserts the UTF-8 encoding of Unicode code point U+0ABC. The braces are mandatory.
- \`\\z\` skips all following whitespace, including newlines, which is useful for splitting a long literal across lines.

\`\`\`luau
local smile = "\\u{1F600}"
local long = "a long line \\z
   continued without the indentation"
\`\`\`

The \`\\u\` form needs braces: \`\\u{41}\`, not \`\\u41\`. Plain \`\\x\` escapes insert raw bytes, so they are not necessarily valid UTF-8.`,
    source: `${SYNTAX}#string-literals`,
    tags: ["syntax", "strings"],
    keywords: ["string escape", "unicode", "hex escape", "\\z", "UTF-8"],
  },
  {
    id: "number-literals",
    questions: [
      "What number formats does Luau support?",
      "Does Luau support binary literals?",
      "How do I use underscores in Luau numbers?",
      "What is the Luau number type?",
      "Does Luau have 64-bit integers?",
      "How do I write a hex number in Luau?",
      "What is the precision of Luau numbers?",
    ],
    answer: `Luau has a single number type: a 64-bit IEEE 754 double. It represents integers exactly up to 2^53, and larger integer literals are stored with precision loss.

Number literals can be decimal, hexadecimal (\`0xABC\` or \`0XABC\`), or binary (\`0b0101\` or \`0B0101\`). Underscores are allowed as visual separators in integer literals:

\`\`\`luau
local million = 1_048_576
local mask = 0xFFFF_FFFF
local bits = 0b_0101_0101
\`\`\`

There is no separate 64-bit integer type and no integer overflow. The linter warns when a hexadecimal or binary literal is truncated, or when a decimal literal cannot be represented exactly.`,
    source: `${SYNTAX}#number-literals`,
    tags: ["syntax", "numbers"],
    keywords: ["number literal", "hexadecimal", "binary", "underscore", "IEEE 754"],
  },
  {
    id: "continue-statement",
    questions: [
      "How do I skip to the next iteration in Luau?",
      "Does Luau have continue?",
      "How does continue work in a Luau loop?",
      "Is continue a keyword in Luau?",
      "Can continue be used in repeat until?",
      "Where must continue appear?",
      "How do I continue a for loop in Luau?",
    ],
    answer: `Luau adds a \`continue\` statement to \`for\`, \`while\`, and \`repeat\` loops. It skips the rest of the current iteration and starts the next one. Like \`break\`, it must be the last statement in its block.

\`\`\`luau
for _, value in values do
    if value < 0 then
        continue
    end
    print(value)
end
\`\`\`

\`continue\` is not a keyword. It is contextual, so existing Lua code that uses \`continue\` as a variable or function name keeps working. In a \`repeat ... until\` loop, \`continue\` cannot skip the declaration of a local variable that is used in the loop condition.`,
    source: `${SYNTAX}#continue-statement`,
    tags: ["syntax", "loops"],
    keywords: ["continue", "loop", "skip iteration", "repeat until"],
  },
  {
    id: "compound-assignment",
    questions: [
      "What compound assignment operators does Luau have?",
      "Does Luau support +=?",
      "Does Luau have ..= for string concatenation?",
      "Are compound assignments expressions?",
      "How does += work with metatables?",
      "What is the difference between = and += in Luau?",
      "Can I use //= in Luau?",
    ],
    answer: `Luau supports \`+=\`, \`-=\`, \`*=\`, \`/=\`, \`//=\`, \`%=\`, \`^=\`, and \`..=\`. They are statements, not expressions, so they do not produce a value.

\`\`\`luau
local total = 0
total += 5
local message = "Luau"
message ..= " types"
\`\`\`

Each form takes one value on the left and one on the right. Function calls on the left are evaluated only once. Compound assignments call the usual arithmetic metamethods and \`__index\`/\`__newindex\` as needed, so custom types do not need special support.`,
    source: `${SYNTAX}#compound-assignments`,
    tags: ["syntax", "operators"],
    keywords: ["compound assignment", "+=", "..=", "//=", "metamethod"],
  },
  {
    id: "const-bindings",
    questions: [
      "What does const do in Luau?",
      "How do I declare an immutable variable in Luau?",
      "Does const make a table immutable?",
      "How do I freeze a table in Luau?",
      "Is const a keyword in Luau?",
      "Can I use const with type annotations?",
      "What is the difference between const and local?",
    ],
    answer: `\`const\` works like \`local\` but prevents reassignment of the binding:

\`\`\`luau
const limit: number = 10
const names = { "Ada", "Grace" }
\`\`\`

The binding is immutable, including inside nested scopes and with compound assignment, but the referenced value is not. A \`const\` table can still have its fields changed. To make the table's contents read-only, use \`table.freeze\`:

\`\`\`luau
const config = table.freeze({ width = 640 })
\`\`\`

\`const\` is a contextual keyword. It is only treated as a keyword where \`local\` is valid, so old code that uses \`const\` as an identifier still works. Type annotations and multi-assignment are supported.`,
    source: `${SYNTAX}#const-bindings`,
    tags: ["syntax", "const", "immutability"],
    keywords: ["const", "immutable", "table.freeze", "binding"],
  },
  {
    id: "if-expressions",
    questions: [
      "How do I write a ternary in Luau?",
      "What is an if-then-else expression?",
      "Does Luau have the question mark colon operator?",
      "How do I return a conditional value in Luau?",
      "Why is if then else better than and or in Luau?",
      "How do I write an inline if in Luau?",
      "Can if expressions have elseif?",
    ],
    answer: `Luau has if-then-else expressions, which return a value and do not use \`end\`:

\`\`\`luau
local label = if score >= 50 then "pass" else "fail"
local size = if n < 0 then "negative" elseif n == 0 then "zero" else "positive"
\`\`\`

The \`else\` branch is mandatory. An if expression can appear anywhere a normal expression can.

Prefer it over the Lua idiom \`a and b or c\`. That idiom returns \`c\` when \`b\` is \`false\` or \`nil\`, even when \`a\` is truthy. The if expression behaves as written in every case. The linter flags misleading and-or expressions for this reason.`,
    source: `${SYNTAX}#if-then-else-expressions`,
    tags: ["syntax", "expressions"],
    keywords: ["if expression", "ternary", "elseif", "and or"],
  },
  {
    id: "generalized-iteration",
    questions: [
      "How do I iterate over a table in Luau?",
      "What is generalized iteration?",
      "Do I need pairs in Luau?",
      "How do I loop over a dictionary in Luau?",
      "What is the iteration order of a Luau table?",
      "Can I use for k, v in t?",
      "What is the difference between pairs and generalized iteration?",
    ],
    answer: `Luau extends \`for ... in\` with generalized iteration, so a table can be iterated directly:

\`\`\`luau
local scores = { alice = 10, bob = 20 }
for name, score in scores do
    print(name, score)
end
\`\`\`

\`pairs(t)\` and \`ipairs(t)\` still work, but generalized iteration is recommended unless you need vanilla Lua compatibility or the exact semantics of \`ipairs\` (which stops at the first \`nil\`).

Iteration visits indices \`1..#t\` consecutively and then the remaining keys in an unspecified order. Changing a value for the current key is fine; changing other keys during iteration gives unspecified behavior. As an exception, for a dense array you may append at \`#t + 1\` during iteration and the new elements are visited in order.`,
    source: `${SYNTAX}#generalized-iteration`,
    tags: ["syntax", "tables", "loops"],
    keywords: ["generalized iteration", "pairs", "ipairs", "table iteration"],
  },
  {
    id: "iter-metamethod",
    questions: [
      "What is the __iter metamethod?",
      "How do I make a custom type iterable in Luau?",
      "How does __iter work?",
      "Can I iterate over userdata in Luau?",
      "How do I implement an iterator for a table?",
      "What should __iter return?",
      "Does Luau support __pairs?",
    ],
    answer: `\`__iter\` lets tables and userdata define their own iteration. Luau calls it before the loop starts, and it must return an iterator function like \`next\` or a custom function:

\`\`\`luau
local range = setmetatable({ n = 3 }, {
    __iter = function(self)
        local i = 0
        return function()
            i += 1
            if i <= self.n then
                return i
            end
        end
    end,
})

for value in range do
    print(value)
end
\`\`\`

\`__iter\` replaces the need for \`__pairs\` and \`__ipairs\`, which Luau does not support. The iterator is called repeatedly until it returns \`nil\` as the first value.`,
    source: `${SYNTAX}#generalized-iteration`,
    tags: ["syntax", "metatables", "iteration"],
    keywords: ["__iter", "iterator", "metamethod", "__pairs", "userdata"],
  },
  {
    id: "string-interpolation",
    questions: [
      "How do I interpolate a string in Luau?",
      "What are backtick strings?",
      "How do I use braces in a Luau string literal?",
      "How do I escape a brace in string interpolation?",
      "Does Luau support template literals?",
      "Can I use expressions in backtick strings?",
      "Why is my interpolated string a parse error?",
      "Can backtick strings be used in type annotations?",
    ],
    answer: `Backtick string literals interpolate expressions written inside \`{ ... }\`:

\`\`\`luau
local name = "Ada"
local score = 42
print(\`Hello {name}, you scored {score + 1}!\`)
\`\`\`

Any expression can appear inside the braces. Inside a backtick literal, \`\\\`\` escapes a backtick, \`\\{\` escapes an opening brace, \`\\\\\` escapes a backslash, and a backslash before a newline escapes the newline.

Two opening braces \`{{\` are rejected with a parse error so that developers do not mistake them for an escaped brace. Backtick literals are not supported in type annotations. Unlike ordinary string literals, a backtick literal used as a function argument must be wrapped in parentheses, because \`f\`text\`\` would otherwise parse as a function call with a string literal.`,
    source: `${SYNTAX}#string-interpolation`,
    tags: ["syntax", "strings"],
    keywords: ["string interpolation", "backtick", "template literal", "{expr}"],
  },
  {
    id: "floor-division",
    questions: [
      "What is // in Luau?",
      "How do I do integer division in Luau?",
      "Does Luau have floor division?",
      "What does //= do?",
      "How does floor division work with vectors?",
      "What happens when you divide by zero with //?",
      "Is floor division the same as math.floor?",
    ],
    answer: `Luau supports floor division with \`//\`, its compound form \`//=\`, and the \`__idiv\` metamethod. For numbers:

\`\`\`luau
print(7 // 2)   -- 3
print(-7 // 2)  -- -4
\`\`\`

\`a // b\` is equivalent to \`math.floor(a / b)\`. If \`b\` is zero, the result is positive or negative infinity; if both are zero, it is NaN. For native vectors, \`//\` applies \`math.floor\` to each component, equivalent to using \`math.floor\` on \`x / d\`, \`y / d\`, and \`z / d\` separately.`,
    source: `${SYNTAX}#floor-division`,
    tags: ["syntax", "operators", "numbers"],
    keywords: ["floor division", "//", "//=", "__idiv", "integer division"],
  },
  {
    id: "attributes",
    questions: [
      "What are attributes in Luau?",
      "What does @native do?",
      "How do I mark a function deprecated in Luau?",
      "Can I define my own attributes?",
      "What attributes does Luau support?",
      "How do I use @deprecated with a reason?",
      "Do attributes apply to nested functions?",
    ],
    answer: `Attributes are built-in annotations placed before a function declaration. They adjust the compiler, analyzer, or runtime. Users cannot define new attributes.

- \`@native\` requests native code generation for that function. It is the function-level form of the \`--!native\` script directive. It does not apply recursively to functions defined inside the annotated function.
- \`@deprecated\` marks a function as deprecated, so the linter warns on calls and the editor shows it differently in autocomplete.

Use the parameterized form for a better message:

\`\`\`luau
@[deprecated { use = "newApi()", reason = "oldApi will be removed" }]
local function oldApi() end
\`\`\`

\`@attr\`, \`@[attr]\`, and \`@[attr()]\` are equivalent. Multiple attributes can share one line.`,
    source: `${ATTRIBUTES}#attributes`,
    tags: ["syntax", "attributes", "performance"],
    keywords: ["attribute", "@native", "@deprecated", "annotation"],
  },
  {
    id: "gradual-typing",
    questions: [
      "What is gradual typing in Luau?",
      "Does Luau require type annotations?",
      "How does Luau type checking work?",
      "What are the benefits of Luau types?",
      "Can I write Luau without types?",
      "What is the Luau type system?",
      "Does Luau compile types away?",
    ],
    answer: `Luau uses a gradual type system: annotations are optional, and the type checker infers what it can. You can start with no annotations and add them where they help.

Types exist for analysis, not runtime. They produce warnings, errors, and autocomplete suggestions; they do not change the generated code. With \`--!strict\`, the checker tracks types across statements and reports code that might fail. With \`--!nonstrict\`, it only reports code that will definitely fail and falls back to \`any\` when inference is uncertain.

The system supports primitives, table shapes, function types, unions, intersections, generics, singleton types, type packs, refinements, and analysis-time type functions.`,
    source: `${TYPES}#gradual-typing`,
    tags: ["types", "overview"],
    keywords: ["gradual typing", "type inference", "annotations", "strict"],
  },
  {
    id: "structural-typing",
    questions: [
      "Is Luau structurally typed?",
      "What does structural typing mean in Luau?",
      "Do Luau types use nominal typing?",
      "Can I pass a table with extra fields?",
      "How does Luau compare table types?",
      "What is width subtyping in Luau?",
      "Why does a table with more fields match a smaller type?",
    ],
    answer: `Luau's type system is structural by default. Two table types are compatible when their shapes are compatible, regardless of names or where they were declared. Lua 5.1 tables are inherently structural, so this matches the language.

The state of a table matters. Unsealed tables, created from literals, are exact and can gain properties. Sealed tables, produced by an explicit annotation or a function return, are inexact: a table with more properties can be used where a type with fewer properties is expected. That is width subtyping. A table with the required fields and extra fields is therefore accepted.`,
    source: `${TYPES}#structural-type-system`,
    tags: ["types", "structural"],
    keywords: ["structural typing", "width subtyping", "sealed table", "shape"],
  },
  {
    id: "type-annotations",
    questions: [
      "How do I annotate a variable in Luau?",
      "How do I type a function argument?",
      "How do I declare a return type?",
      "What is the syntax for type annotations?",
      "How do I annotate a local variable?",
      "How do I type multiple return values?",
      "How do I annotate a table variable?",
    ],
    answer: `Put a colon and a type after a local, parameter, or return type:

\`\`\`luau
local count: number = 0
local names: { string } = { "Ada", "Grace" }

local function greet(name: string, times: number): string
    return string.rep("Hello " .. name .. " ", times)
end
\`\`\`

For multiple return values, wrap the list in parentheses:

\`\`\`luau
local function pair(): (number, string)
    return 1, "one"
end
\`\`\`

Function types are written with the argument list, an arrow, and the return type, for example \`(number, string) -> boolean\`. Table types use the table-literal shape, and \`{T}\` is shorthand for an array-like table.`,
    source: `${SYNTAX}#type-annotations`,
    tags: ["types", "syntax"],
    keywords: ["type annotation", "parameter", "return type", "local"],
  },
  {
    id: "type-casts",
    questions: [
      "How do I cast a type in Luau?",
      "What does :: do in Luau?",
      "How do I use the :: operator?",
      "Can I cast any type in Luau?",
      "Why do I need a type cast?",
      "What happens to multiple return values when cast?",
      "How do I cast to any to break a require cycle?",
    ],
    answer: `The \`::\` operator overrides the inferred type of an expression:

\`\`\`luau
local names = {} :: { string }
names[1] = "Ada"
\`\`\`

Casts are checked. One of the two types must be a subtype of the other, or the cast must involve \`any\`, so \`string :: number\` is rejected. Use a cast when inference is too broad, not to bypass an unsafe conversion.

Casting a variadic expression or a call with multiple returns keeps only the first value. Casting a module to \`any\` is also the documented escape hatch for breaking a cyclic module dependency.`,
    source: `${TYPES}#type-casts`,
    tags: ["types", "casts"],
    keywords: ["type cast", "::", "typecast", "subtype"],
  },
  {
    id: "type-aliases",
    questions: [
      "How do I create a type alias in Luau?",
      "What is the type keyword?",
      "How do I export a type from a module?",
      "Can a type alias use typeof?",
      "Are type aliases global?",
      "How do I import an exported type?",
      "What is the syntax for type Foo equals?",
    ],
    answer: `Declare a type alias with \`type\`:

\`\`\`luau
type Point = { x: number, y: number }
type Callback = (value: string) -> boolean
\`\`\`

The right-hand side can be a type or a \`typeof(expression)\`; the expression is not evaluated at runtime. Aliases are local to the file by default. To share one, export it and import it through the module alias:

\`\`\`luau
-- shapes.luau
export type Point = { x: number, y: number }

-- main.luau
local shapes = require("./shapes")
local origin: shapes.Point = { x = 0, y = 0 }
\`\`\`

Exported types are used by prefixing the name with the require alias.`,
    source: `${SYNTAX}#type-annotations`,
    tags: ["types", "aliases"],
    keywords: ["type alias", "export type", "type keyword", "typeof"],
  },
  {
    id: "typeof-types",
    questions: [
      "What is typeof in Luau?",
      "How do I get the type of a value?",
      "Does typeof run at runtime?",
      "How do I type a metatable-based class?",
      "What is the difference between type and typeof?",
      "Can I use typeof in a type alias?",
      "How do I get the type of a module?",
    ],
    answer: `There are two related tools. In a type position, \`typeof(expression)\` produces the static type of that expression, and the expression is not evaluated at runtime. At runtime, the global function \`typeof(value)\` returns a type name as a string; for host-defined userdata with a \`__type\` field it returns that value.

\`typeof\` is how metatable-based classes are typed:

\`\`\`luau
local Account = {}
Account.__index = Account

export type Account = typeof(setmetatable(
    {} :: { name: string, balance: number },
    Account
))
\`\`\`

The runtime \`typeof\` is like \`type\`, but it can distinguish host-defined userdata and is not spoofable by \`newproxy\`.`,
    source: `${SYNTAX}#type-annotations`,
    tags: ["types", "typeof", "oop"],
    keywords: ["typeof", "type alias", "runtime type", "class"],
  },
  {
    id: "primitive-types",
    questions: [
      "What are the primitive types in Luau?",
      "How many types does the Luau VM have?",
      "What is the vector type?",
      "What is the buffer type?",
      "Does Luau have userdata?",
      "What are the builtin types?",
      "How do I represent a function type?",
      "Is table a primitive type?",
    ],
    answer: `The Luau VM has ten primitive types: \`nil\`, \`string\`, \`number\`, \`boolean\`, \`table\`, \`function\`, \`thread\`, \`userdata\`, \`vector\`, and \`buffer\`.

Some have dedicated syntax rather than a name. Tables use table type syntax such as \`{ [string]: number }\`. Functions use argument and return types such as \`(number) -> string\`. \`userdata\` is represented by concrete types supplied by the embedder. \`vector\` is not representable by name in an annotation.

The type checker also provides \`unknown\`, \`never\`, and \`any\`, which are analysis types rather than VM runtime tags.`,
    source: `${BASIC_TYPES}#builtin-types`,
    tags: ["types", "primitives"],
    keywords: ["primitive types", "vector", "buffer", "userdata", "thread"],
  },
  {
    id: "any-unknown-never",
    questions: [
      "What is the difference between any and unknown in Luau?",
      "What is the never type?",
      "What is the unknown type?",
      "What does any do in Luau?",
      "When should I use unknown instead of any?",
      "How do I use a value of type unknown?",
      "What is the top type in Luau?",
      "What is the bottom type in Luau?",
    ],
    answer: `\`any\` is an opt-out from the type system. A value typed \`any\` can be used as any other type without further checks. It is convenient, but it hides mistakes.

\`unknown\` is the top type: it is a union of every type. Unlike \`any\`, it cannot be used as a more specific type until it has been refined. Use \`unknown\` for values whose shape you do not trust and refine with \`type\`, equality, or truthiness checks before using them.

\`never\` is the bottom type. No value inhabits it. It is the dual of \`unknown\` and is mainly useful when type refinements prove that a branch is impossible.`,
    source: `${BASIC_TYPES}#unknown-type`,
    tags: ["types", "any", "unknown", "never"],
    keywords: ["any", "unknown", "never", "top type", "bottom type"],
  },
  {
    id: "singleton-types",
    questions: [
      "What are singleton types in Luau?",
      "Does Luau support literal types?",
      "Can I use a string literal as a type?",
      "Can I use true and false as types?",
      "Does Luau support number literal types?",
      "How do I restrict a string to specific values?",
      "What is a literal type?",
    ],
    answer: `Singleton types, also called literal types, represent exactly one runtime value. Luau currently supports string and boolean singletons:

\`\`\`luau
type Direction = "north" | "south" | "east" | "west"
type Always = true
\`\`\`

String singletons are commonly used as tags in tagged unions and as the refined type produced by an equality check. Boolean singletons are useful for enforcing invariants.

Number literal types are not currently supported. This is intentional rather than an oversight. Singletons also appear naturally in type refinements: \`if value == "hello" then\` refines \`value\` to the singleton type \`"hello"\`.`,
    source: `${BASIC_TYPES}#singleton-types-aka-literal-types`,
    tags: ["types", "singletons"],
    keywords: ["singleton type", "literal type", "string literal", "boolean literal"],
  },
  {
    id: "function-types",
    questions: [
      "How do I write a function type in Luau?",
      "What is the syntax for a callback type?",
      "How do I type a function with no return values?",
      "How do I type multiple return values?",
      "Can function types name their parameters?",
      "What is the function type arrow syntax?",
      "How do I type a variadic function?",
    ],
    answer: `A function type lists the parameters, an arrow, and the return type:

\`\`\`luau
type Callback = (value: string, index: number) -> boolean
type Nothing = () -> ()
type Pair = (number) -> (number, string)
\`\`\`

Parameter names are optional and only show up in documentation and autocomplete. To return no values, or more than one, wrap the return list in parentheses. A variadic parameter is written with \`...\`:

\`\`\`luau
type Sum = (...number) -> number
\`\`\`

Function types can also use generics, unions, and intersections, for example \`((number) -> string) & ((boolean) -> string)\` for an overloaded builtin.`,
    source: `${BASIC_TYPES}#function-types`,
    tags: ["types", "functions"],
    keywords: ["function type", "callback", "arrow", "variadic", "return type"],
  },
  {
    id: "variadic-type-packs",
    questions: [
      "What is a type pack in Luau?",
      "How do I type ellipsis in Luau?",
      "What is the difference between ...T and U...?",
      "Can a generic function take a type pack?",
      "How do I declare a generic type pack?",
      "How do I instantiate a type with a type pack?",
    ],
    answer: `A type pack represents a list of types. Multiple return values and a variadic parameter use packs.

- \`...T\` is a variadic type pack: zero or more values that all have type \`T\`.
- \`U...\` is a generic type pack: zero or more types that do not have to match.

A generic type alias or function can declare a pack parameter after its type parameters:

\`\`\`luau
type Callback<A..., R...> = (A...) -> R...

local function call<A..., R...>(fn: (A...) -> R..., ...: A...): R...
    return fn(...)
end
\`\`\`

Packs can be instantiated with a parenthesized list such as \`f<(number, string)>()\`, or by passing trailing type arguments. A generic function can declare more than one pack if needed.`,
    source: `${BASIC_TYPES}#type-packs`,
    tags: ["types", "type packs", "variadic"],
    keywords: ["type pack", "variadic", "...T", "generic pack"],
  },
  {
    id: "table-types",
    questions: [
      "What are the three table states in Luau types?",
      "What is an unsealed table?",
      "What is a sealed table?",
      "What is a generic table type?",
      "Why does my table type reject a new field?",
      "What does exact mean for Luau tables?",
      "How does width subtyping work?",
    ],
    answer: `From the type checker's perspective, a table is unsealed, sealed, or generic.

- An unsealed table comes from a table literal. It can gain properties, and the type updates. It is exact: every property must be named by the type, though missing properties are treated as optional \`nil\`.
- A sealed table is locked down by an explicit annotation or by being returned from a function. It is inexact and supports width subtyping, so a table with more properties can stand in for a type with fewer.
- A generic table is a table parameter or value with no concrete shape yet. Indexing it requests a table with a matching interface.

\`\`\`luau
local t = { x = 1 }       -- unsealed, can add fields
local fixed: { x: number } = { x = 1 } -- sealed, cannot add fields
\`\`\`

Once the scope that created an unsealed table ends, the table seals.`,
    source: `${TABLE_TYPES}#unsealed-tables`,
    tags: ["types", "tables"],
    keywords: ["unsealed table", "sealed table", "generic table", "exact", "width subtyping"],
  },
  {
    id: "table-indexers",
    questions: [
      "How do I type a dictionary in Luau?",
      "What is a table indexer type?",
      "How do I write an array type in Luau?",
      "What does array-of-string mean in Luau types?",
      "How do I type a table with numeric keys?",
      "How do I type a table with mixed fields and an indexer?",
      "What is the syntax for an index signature?",
    ],
    answer: `Use an indexer to describe keys and values:

\`\`\`luau
type Scores = { [string]: number }
type Names = { [number]: string }
type NamesArray = { string } -- same as { [number]: string }
\`\`\`

The compact \`{T}\` form is the array-like shorthand. The explicit indexer form is useful when the key is not a number, or when the table also has named fields:

\`\`\`luau
type Inventory = {
    [string]: number,
    count: number,
}
\`\`\`

Indexers can have separate read and write types in type-function APIs, but ordinary annotations use one type for both. A table can have an array part and a hash part at runtime, and the type describes both.`,
    source: `${TABLE_TYPES}#table-indexers`,
    tags: ["types", "tables", "indexers"],
    keywords: ["table indexer", "dictionary", "array type", "index signature"],
  },
  {
    id: "unions-and-optional",
    questions: [
      "How do I write a union type in Luau?",
      "What does the pipe operator mean in Luau types?",
      "How do I make a type optional?",
      "What does a question mark mean in a type?",
      "Can I call a union of functions?",
      "How do I represent string or number?",
      "Is number optional the same as number or nil?",
    ],
    answer: `A union type means a value is one of several types:

\`\`\`luau
type Id = string | number
type MaybeName = string?
\`\`\`

\`T?\` is shorthand for \`T | nil\`. It is extremely common for optional function arguments and table fields.

You must narrow a union before using a member-specific operation. For example, check \`type(value) == "number"\` before doing arithmetic on \`string | number\`. A union of two or more function types cannot be called, because the checker cannot know which signature applies. \`any\` is technically a union of every type.`,
    source: `${UNIONS}#union-types`,
    tags: ["types", "unions"],
    keywords: ["union type", "optional", "pipe", "question mark", "nil"],
  },
  {
    id: "intersection-types",
    questions: [
      "What is an intersection type in Luau?",
      "What does the ampersand mean in Luau types?",
      "How do I combine two table types?",
      "Can I intersect primitives in Luau?",
      "What are intersection types used for?",
      "How do I type an overloaded function?",
      "Does Luau support user-defined overloads?",
    ],
    answer: `An intersection type requires a value to satisfy all of the listed types at once:

\`\`\`luau
type Named = { name: string }
type Aged = { age: number }
type Person = Named & Aged
\`\`\`

Intersections are mainly used to combine table shapes and to describe overloadable function signatures, such as a function that accepts either a number or a boolean:

\`\`\`luau
type Describe = ((number) -> string) & ((boolean) -> string)
\`\`\`

Intersecting incompatible primitives, such as \`string & number\`, is impossible and rejected. Luau does not allow users to define their own overloaded functions; intersections are for builtin and host APIs that genuinely have multiple signatures.`,
    source: `${UNIONS}#intersection-types`,
    tags: ["types", "intersections"],
    keywords: ["intersection type", "ampersand", "overload", "combine tables"],
  },
  {
    id: "tagged-unions",
    questions: [
      "What is a tagged union in Luau?",
      "How do I discriminate a union by a type field?",
      "How do I model a Result type in Luau?",
      "How do I narrow a union table?",
      "What is a discriminated union?",
      "Why does my union field not exist?",
      "How do I use type refinements with tagged unions?",
    ],
    answer: `A tagged union is a union of tables that share a literal tag field:

\`\`\`luau
type Result<T, E> =
    { type: "ok", value: T }
    | { type: "err", error: E }

local function unwrap<T, E>(result: Result<T, E>): T
    if result.type == "ok" then
        return result.value
    else
        error(result.error)
    end
end
\`\`\`

Refining on \`result.type\` narrows the union to one variant. That is why \`result.value\` is only available when \`type\` is \`"ok"\`, and \`result.error\` only when it is \`"err"\`. The tag must be a singleton type such as a string literal for the refinement to work.`,
    source: `${UNIONS}#tagged-unions`,
    tags: ["types", "unions", "refinements"],
    keywords: ["tagged union", "discriminated union", "Result", "tag", "narrow"],
  },
  {
    id: "generics",
    questions: [
      "How do I write a generic function in Luau?",
      "What is a type parameter?",
      "How does Luau infer generic types?",
      "How do I type a function that returns the same type?",
      "Does Luau support generics?",
      "Can generic functions have defaults?",
      "How do I declare a generic type parameter?",
    ],
    answer: `A generic is a type parameter that another type is slotted into. Declare parameters in angle brackets before the function:

\`\`\`luau
local function first<T>(values: { T }): T?
    return values[1]
end
\`\`\`

At the call site, Luau infers the type arguments. \`first({ "a", "b" })\` infers \`T = string\`, and the result is \`string?\`. Generics preserve the actual type instead of erasing it to \`any\`.

Generic type aliases work too, such as \`type Pair<T> = { first: T, second: T }\`. Builtins use generics as well; the type of two-argument \`table.insert\` is \`<T>({T}, T) -> ()\`.

Generic type parameters cannot have defaults.`,
    source: `${GENERICS}#generic-functions`,
    tags: ["types", "generics"],
    keywords: ["generic function", "type parameter", "polymorphism", "<T>"],
  },
  {
    id: "generic-instantiation",
    questions: [
      "How do I explicitly instantiate a generic in Luau?",
      "What is the double angle bracket syntax?",
      "How do I specify a type argument manually?",
      "Why would I use explicit type instantiation?",
      "Can I instantiate multiple type parameters?",
      "How do I constrain a generic call?",
    ],
    answer: `Generics are inferred by default, but you can provide type arguments explicitly with double angle brackets:

\`\`\`luau
local function identity<T>(value: T): T
    return value
end

local n = identity<<number>>(42)
\`\`\`

Multiple parameters are separated by commas: \`f<<string, number>>("a", 1)\`. Explicit instantiation is useful when inference would be ambiguous, or when you want to constrain an argument to a supertype instead of the exact inferred type.

The syntax is \`func<<Type>>(...)\`, not \`func<Type>(...)\`; single angle brackets remain comparison operators.`,
    source: `${GENERICS}#generic-type-instantiation`,
    tags: ["types", "generics"],
    keywords: ["generic instantiation", "<<>>", "type argument", "explicit"],
  },
  {
    id: "type-refinements",
    questions: [
      "What is a type refinement in Luau?",
      "How do I narrow a type in Luau?",
      "How does if x then narrow a type?",
      "How do I use type of x equals number?",
      "Does assert refine types?",
      "How do I narrow with equality?",
      "Can I combine refinements with and or not?",
      "Why is my nil check not narrowing?",
    ],
    answer: `A type refinement narrows the type of an lvalue (a global, local, or property). Luau supports:

- Truthy tests: \`if value then\` removes \`false\` and \`nil\`.
- Type guards: \`if type(value) == "number" then\` refines to \`number\`.
- Equality: \`if value == "hello" then\` refines to the singleton \`"hello"\`.
- \`assert(value, "message")\`, which refines the same way as an if test.

Refinements compose with \`and\`, \`or\`, and \`not\`. \`not\` and \`~=\` flip the refinement, so \`not value\` refines to a falsy type. If a check does not narrow, make sure it is on the exact lvalue the checker tracks and that the comparison uses a literal or a recognized type guard.`,
    source: `${REFINEMENTS}#type-refinements`,
    tags: ["types", "refinements", "narrowing"],
    keywords: ["type refinement", "narrow", "type guard", "truthy", "assert"],
  },
  {
    id: "oop-typing",
    questions: [
      "How do I type an object-oriented class in Luau?",
      "How do I type self in Luau?",
      "How do I annotate a metatable class?",
      "Why is self not inferred across methods?",
      "How do I define a class type with setmetatable?",
      "How do I type a constructor like Account.new?",
      "Does Luau support classes?",
    ],
    answer: `Luau has no dedicated class syntax. The common pattern is a table with a metatable. The type checker can infer part of it, but \`self\` is not shared across methods, because Luau allows calling a method with an explicit different \`self\`:

\`\`\`luau
local Account = {}
Account.__index = Account

export type Account = typeof(setmetatable(
    {} :: { name: string, balance: number },
    Account
))

function Account.new(name: string, balance: number): Account
    return setmetatable({ name = name, balance = balance }, Account)
end

function Account.deposit(self: Account, amount: number)
    self.balance += amount
end
\`\`\`

Annotate the data shape first, then define the class type with \`typeof(setmetatable(...))\`, or the \`setmetatable\` type function. After that, annotate each method's \`self\` explicitly. You can still call \`account:deposit(50)\`. The Luau team plans shared self types for functions defined with colon syntax.`,
    source: `${OOP}#adding-types-for-faux-object-oriented-programs`,
    tags: ["types", "oop", "metatables"],
    keywords: ["class", "self", "metatable", "setmetatable", "Account"],
  },
  {
    id: "type-functions-and-types-library",
    questions: [
      "What is a type function in Luau?",
      "What is the types library?",
      "How do I write a type function?",
      "When do type functions run?",
      "What is keyof in Luau?",
      "What can type functions access?",
      "How do I create a union type in a type function?",
      "What is types.singleton?",
    ],
    answer: `Type functions run during analysis and operate on types, not runtime values. They use the \`types\` library to inspect and build types:

\`\`\`luau
type function keyof(ty)
    assert(ty:is("table"), "keyof expects a table type")
    local names = {}
    for name in ty:properties() do
        table.insert(names, name)
    end
    return types.unionof(table.unpack(names))
end
\`\`\`

This is a simplified form of the builtin \`keyof\`. Type functions can create singletons, optional types, unions, intersections, table types, function types, generic types, and copies.

Besides \`types\`, they can use \`assert\`, \`error\`, \`print\`, \`next\`, \`ipairs\`, \`pairs\`, \`select\`, \`unpack\`, \`getmetatable\`, \`setmetatable\`, \`rawget\`, \`rawset\`, \`rawlen\`, \`raweq\`, \`tonumber\`, \`tostring\`, \`type\`, \`typeof\`, and the \`math\`, \`table\`, \`string\`, \`bit32\`, \`utf8\`, and \`buffer\` libraries.`,
    source: `${TYPE_FUNCTIONS}#type-functions`,
    tags: ["types", "type functions"],
    keywords: ["type function", "types library", "keyof", "analysis time", "types.singleton"],
  },
  {
    id: "require-typing",
    questions: [
      "How does Luau type require?",
      "How do I type a module import?",
      "What is a cyclic module dependency?",
      "How do I break a require cycle in Luau?",
      "Why can't Luau type check my require?",
      "Does require need a static path?",
      "How do I export types from a module?",
    ],
    answer: `Luau resolves \`require\` paths statically. \`require("./bar")\` means the sibling module \`bar\` relative to the current script. If the path cannot be resolved statically, the type checker cannot analyze the module.

A module's returned table provides its type. Exported types are accessed through the require alias:

\`\`\`luau
local shapes = require("./shapes")
local point: shapes.Point = { x = 1, y = 2 }
\`\`\`

Cyclic module dependencies can confuse the checker. The documented workaround is to cast the module to \`any\` at one point in the cycle, which breaks the analysis dependency. Prefer restructuring to avoid the cycle when possible.`,
    source: `${CONSIDERATIONS}#module-interactions`,
    tags: ["types", "modules", "require"],
    keywords: ["require", "module", "cyclic dependency", "import", "export type"],
  },
  {
    id: "math-library",
    questions: [
      "What math functions does Luau add?",
      "How do I clamp a number in Luau?",
      "How do I round a number in Luau?",
      "What does math.sign do?",
      "How do I interpolate between numbers?",
      "How do I map a number range?",
      "How do I generate Perlin noise in Luau?",
      "What is the rounding rule for math.round?",
    ],
    answer: `Luau adds several helpers to the standard \`math\` library:

- \`math.clamp(n, min, max)\` returns \`n\` limited to the range. It errors if \`min > max\`.
- \`math.sign(n)\` returns \`-1\`, \`0\`, or \`1\`. It returns \`0\` for zero and NaN.
- \`math.round(n)\` rounds to the nearest integer. Exact halfway values round away from zero.
- \`math.lerp(a, b, t)\` interpolates linearly; at \`t == 1\` it returns exactly \`b\`.
- \`math.map(x, inmin, inmax, outmin, outmax)\` remaps a value between ranges.
- \`math.noise(x, y, z)\` returns 3D Perlin noise in \`[-1, 1]\`.
- \`math.isnan\`, \`math.isinf\`, and \`math.isfinite\` test special values.

\`\`\`luau
local hp = math.clamp(hp + 10, 0, 100)
local direction = math.sign(targetX - x)
local shaded = math.lerp(0.2, 1.0, lightLevel)
\`\`\`

\`math.log\` accepts an optional base, and \`math.max\`/\`math.min\` require at least one argument.`,
    source: `${LIBRARY}#math-library`,
    tags: ["library", "math"],
    keywords: ["math.clamp", "math.sign", "math.round", "math.lerp", "math.map", "noise"],
  },
  {
    id: "table-library",
    questions: [
      "What table functions does Luau add?",
      "How do I create a table with preallocated space?",
      "How do I find an element in a table?",
      "How do I clear a table without reallocating?",
      "What does table.move do?",
      "How do I pack varargs into a table?",
      "What is table.create?",
      "What is table.find?",
    ],
    answer: `Luau extends the \`table\` library with allocation and lookup helpers:

- \`table.create(n, value?)\` creates a table with \`n\` array slots, optionally filled with \`value\`. Without a value it preallocates capacity for later inserts.
- \`table.find(t, value, init?)\` returns the first matching index, or \`nil\`.
- \`table.clear(t)\` removes every element while keeping capacity.
- \`table.move(a, from, to, dest, target?)\` copies a range, optionally into another table.
- \`table.pack(...)\` stores arguments plus an \`n\` field; \`table.unpack(t, i, j)\` expands a range.
- \`table.maxn(t)\` returns the largest numeric key, or zero.

\`\`\`luau
local squares = table.create(100)
for i = 1, 100 do
    squares[i] = i * i
end
local index = table.find(squares, 49)
\`\`\`

The classic \`insert\`, \`remove\`, \`sort\`, and \`concat\` functions are also available.`,
    source: `${LIBRARY}#table-library`,
    tags: ["library", "tables"],
    keywords: ["table.create", "table.find", "table.clear", "table.move", "table.pack"],
  },
  {
    id: "table-freeze-clone",
    questions: [
      "How do I make a table read-only in Luau?",
      "What does table.freeze do?",
      "How do I clone a table in Luau?",
      "Is table.clone deep or shallow?",
      "How do I check if a table is frozen?",
      "Can I freeze a table with a protected metatable?",
      "Does table.freeze copy the table?",
    ],
    answer: `\`table.freeze(t)\` freezes a table in place. Any later attempt to modify it, or to assign its metatable, raises an error. \`table.isfrozen(t)\` reports the state. Freezing an already frozen table, or a table with a protected metatable, is an error.

\`\`\`luau
local config = table.freeze({ width = 640, height = 360 })
print(table.isfrozen(config)) -- true
\`\`\`

Freezing is shallow. Nested tables are not frozen automatically. \`table.clone(t)\` returns a shallow copy with the same keys, values, and metatable. The copy is not frozen even if the original was. A deep copy is not automatic because different programs need different rules for cycles and shared references.`,
    source: `${LIBRARY}#table-library`,
    tags: ["library", "tables", "immutability"],
    keywords: ["table.freeze", "table.isfrozen", "table.clone", "read-only", "shallow copy"],
  },
  {
    id: "string-split",
    questions: [
      "How do I split a string in Luau?",
      "What is string.split?",
      "What is the default separator for string.split?",
      "How do I split a string into characters?",
      "How do I split a comma-separated string?",
      "Does standard Lua have string.split?",
      "What does string.split return?",
    ],
    answer: `\`string.split(s, separator?)\` splits a string and returns an array of substrings. The separator defaults to a comma.

\`\`\`luau
local parts = string.split("a,b,c")      -- { "a", "b", "c" }
local path = string.split("a/b/c", "/")  -- { "a", "b", "c" }
local bytes = string.split("abc", "")    -- { "a", "b", "c" }
\`\`\`

An empty separator splits the input into separate one-byte strings. This function is a Luau addition, not part of vanilla Lua. For pattern-based splitting, use \`string.gmatch\` with a pattern.`,
    source: `${LIBRARY}#string-library`,
    tags: ["library", "strings"],
    keywords: ["string.split", "split", "separator", "comma"],
  },
  {
    id: "buffer-library",
    questions: [
      "What is a buffer in Luau?",
      "How do I read binary data in Luau?",
      "How do I create a buffer?",
      "What is the buffer size limit?",
      "How do I read a float from a buffer?",
      "What byte order does buffer use?",
      "How do I copy a buffer?",
      "What is buffer.readbits?",
    ],
    answer: `A \`buffer\` is a fixed-size mutable block of memory. It is useful for binary data, networking, and serialization.

\`\`\`luau
local b = buffer.create(8)
buffer.writef32(b, 0, 1.5)
local value = buffer.readf32(b, 0)
\`\`\`

The library includes \`create\`, \`fromstring\`, \`tostring\`, and \`len\`; signed and unsigned 8/16/32-bit integer reads and writes; \`readf32\`/\`writef32\` and \`readf64\`/\`writef64\`; \`readstring\`/\`writestring\`; \`readbits\`/\`writebits\`; and \`copy\`/\`fill\`.

All multi-byte reads and writes use little-endian order, and integers use two's complement. Out-of-range types take the less significant bits, but keeping values in range is recommended. Reading or writing outside the buffer raises an error. The maximum buffer size is 1 GiB (1,073,741,824 bytes).`,
    source: `${LIBRARY}#buffer-library`,
    tags: ["library", "buffer", "binary"],
    keywords: ["buffer", "binary", "readf32", "writef32", "little endian", "1 GiB"],
  },
  {
    id: "vector-library",
    questions: [
      "What is the vector type in Luau?",
      "How do I create a vector?",
      "How do I normalize a vector?",
      "How do I compute a dot product?",
      "How do I compute a cross product?",
      "Are Luau vectors mutable?",
      "How do I get the magnitude of a vector?",
      "How do I clamp or floor a vector?",
    ],
    answer: `\`vector\` is a native immutable value type with three single-precision components (\`x\`, \`y\`, \`z\`), plus \`w\` when the embedder enables 4-wide mode. The VM implements vector arithmetic directly.

\`\`\`luau
local a = vector.create(1, 0, 0)
local b = vector.create(0, 1, 0)
local up = vector.cross(a, b)
local product = vector.dot(a, b)
local direction = vector.normalize(vector.create(3, 4, 0))
local length = vector.magnitude(direction)
\`\`\`

The library also provides \`angle\`, \`floor\`, \`ceil\`, \`abs\`, \`sign\`, \`clamp\`, \`max\`, \`min\`, and the constants \`vector.zero\` and \`vector.one\`. Components can be read with lowercase or uppercase field names. Because vectors are immutable, assigning a component is not supported.`,
    source: `${LIBRARY}#vector-library`,
    tags: ["library", "vector", "math"],
    keywords: ["vector", "normalize", "dot product", "cross product", "magnitude", "immutable"],
  },
  {
    id: "bit32-library",
    questions: [
      "What is the bit32 library?",
      "How do I do bitwise operations in Luau?",
      "How do I shift bits in Luau?",
      "How do I count leading zeros?",
      "What does bit32.extract do?",
      "How do I replace bits in a number?",
      "Does Luau have bitwise operators?",
    ],
    answer: `\`bit32\` treats numbers as 32-bit unsigned integers in the range \`[0, 4294967295]\`, with bit 0 as the least significant bit. Luau deliberately has no bitwise operators; use this library instead.

It provides \`band\`, \`bor\`, \`bxor\`, \`bnot\`, and \`btest\` for logic; \`lshift\`, \`rshift\`, \`arshift\`, \`lrotate\`, and \`rrotate\` for shifts and rotations; \`extract\` and \`replace\` for bit fields; \`countlz\` and \`countrz\` for zero counts; and \`byteswap\`.

\`\`\`luau
local flags = bit32.bor(1, 4)         -- 5
local low = bit32.extract(0xABCD, 0, 8) -- 0xCD
local shifted = bit32.lshift(1, 4)    -- 16
\`\`\`

Shifts outside the valid range return zero, and arithmetic shift propagates the sign bit. \`extract\` and \`replace\` error when the field is outside bits 0-31.`,
    source: `${LIBRARY}#bit32-library`,
    tags: ["library", "bit32", "bits"],
    keywords: ["bit32", "bitwise", "shift", "extract", "countlz", "byteswap"],
  },
  {
    id: "utf8-library",
    questions: [
      "How do I work with UTF-8 strings in Luau?",
      "What does utf8.len do?",
      "How do I iterate Unicode codepoints?",
      "How do I convert a codepoint to a string?",
      "How do I get the codepoint at a byte offset?",
      "What happens if a UTF-8 string is malformed?",
      "What does utf8.offset do?",
    ],
    answer: `Strings are byte sequences. The \`utf8\` library inspects and builds UTF-8 encoded text:

- \`utf8.len(s, i?, j?)\` returns the number of code points, or \`nil\` plus the first invalid byte position for malformed input.
- \`utf8.codepoint(s, i?, j?)\` returns the code points in a byte range.
- \`utf8.char(...)\` builds a string from code points.
- \`utf8.offset(s, n, i?)\` returns the byte offset of the \`n\`th code point.
- \`utf8.codes(s)\` returns an iterator of byte offset and code point.

\`\`\`luau
for offset, code in utf8.codes("héllo") do
    print(offset, code)
end

local text = utf8.char(0x1F600)
\`\`\`

\`#s\` and \`string.len\` count bytes, not characters. Use \`utf8.len\` when you need character counts. Invalid byte sequences do not throw from every function; check the documented \`nil\` result where it applies.`,
    source: `${LIBRARY}#utf8-library`,
    tags: ["library", "utf8", "strings"],
    keywords: ["utf8", "codepoint", "UTF-8", "utf8.len", "utf8.codes"],
  },
  {
    id: "coroutine-library",
    questions: [
      "How do I create a coroutine in Luau?",
      "What does coroutine.close do?",
      "What is coroutine.finally?",
      "How do I check if a coroutine can yield?",
      "What is coroutine.isyieldable?",
      "How do I resume a coroutine?",
      "What is the difference between coroutine.wrap and create?",
    ],
    answer: `Luau keeps the standard coroutine functions: \`create\`, \`resume\`, \`yield\`, \`status\`, and \`wrap\`. A coroutine created with \`create\` returns \`true\` plus values from \`resume\`, or \`false\` plus the error. \`wrap\` returns a function that resumes the coroutine for you and propagates errors.

Luau adds:

- \`coroutine.close(co)\` puts a dead or suspended coroutine into the dead state and clears its stack. It cannot close a running coroutine.
- \`coroutine.isyieldable()\` reports whether the current coroutine can yield.
- \`coroutine.finally(co, callback)\` registers a callback that receives \`"finished"\`, \`"error"\`, or \`"cancelled"\` when the coroutine ends.

\`pcall\` and \`xpcall\` are yieldable in Luau, but yielding inside most metamethods is not supported. \`coroutine.close\` corresponds to Lua 5.4's \`lua_resetthread\`.`,
    source: `${LIBRARY}#coroutine-library`,
    tags: ["library", "coroutines"],
    keywords: ["coroutine", "close", "finally", "isyieldable", "resume", "yield"],
  },
  {
    id: "os-debug-library",
    questions: [
      "What os functions are available in Luau?",
      "Does Luau have os.time?",
      "Does Luau have os.date?",
      "What debug functions does Luau support?",
      "Is os.execute available?",
      "How do I measure time in Luau?",
      "What does debug.info do?",
      "Is os.remove available?",
    ],
    answer: `Luau keeps a small, sandbox-friendly subset of \`os\`:

- \`os.clock()\` returns a high-precision duration counter in seconds.
- \`os.time(table?)\` returns a Unix timestamp; with a table it uses UTC.
- \`os.date(format?, time?)\` formats a date; \`!*t\` returns a table.
- \`os.difftime(a, b)\` is provided for compatibility; prefer \`a - b\`.

There is no \`os.execute\`, \`os.exit\`, \`os.remove\`, \`os.rename\`, \`os.tmpname\`, \`os.getenv\`, or \`os.setlocale\`.

\`debug\` is also limited to \`debug.info\` (source, line, name, function, and arity information) and \`debug.traceback\`. Luau does not support debug hooks; breakpoints and stepping are implemented by the debugger through bytecode patching and a custom interpreter loop, so they do not slow normal execution.`,
    source: `${LIBRARY}#os-library`,
    tags: ["library", "os", "debug"],
    keywords: ["os.time", "os.date", "os.clock", "debug.info", "debug.traceback"],
  },
  {
    id: "missing-libraries",
    questions: [
      "Does Luau have the io library?",
      "Does Luau support files?",
      "Can I use loadstring in Luau?",
      "Is the package library available?",
      "Why is string.dump missing?",
      "Does Luau have dofile or loadfile?",
      "How do I require a file in Luau?",
      "Is newproxy limited?",
    ],
    answer: `Luau removes several Lua 5.1 facilities for sandboxing and portability:

- The \`io\` and \`package\` libraries are not available, so scripts have no direct file or package access.
- \`loadfile\` and \`dofile\` are removed.
- \`loadstring\` exists but cannot load precompiled bytecode.
- \`string.dump\` is removed because exposing bytecode is unsafe for sandboxes.
- \`newproxy\` only accepts \`nil\` or a boolean, not a custom metatable.

A host application supplies its own module system and APIs. In Roblox, for example, \`require\` loads ModuleScripts. In the standalone CLI, the environment decides what is available.

Some \`os\` and \`debug\` functions remain, as documented separately. This is why Luau is described as based on Lua 5.1 rather than a complete copy of it.`,
    source: `${COMPATIBILITY}#lua-51`,
    tags: ["compatibility", "sandbox", "libraries"],
    keywords: ["io", "package", "loadstring", "string.dump", "loadfile", "dofile", "newproxy"],
  },
  {
    id: "native-codegen",
    questions: [
      "How do I enable native code generation in Luau?",
      "What does --!native do?",
      "What is the @native attribute?",
      "Does Luau have a JIT?",
      "Which platforms support native Luau?",
      "Does @native apply to nested functions?",
      "When should I use native code?",
    ],
    answer: `Luau has an optional native code generation path and a JIT for x64 and arm64. Native compilation is opt-in:

- Add \`--!native\` at the top of a script to request it for that script.
- Add \`@native\` before a function to request it for that function. It is the function-level equivalent and does not apply recursively to nested functions.

\`\`\`luau
--!native

@native
local function expensive(x: number): number
    return math.sqrt(x) * 2
end
\`\`\`

Native code is most useful for compute-heavy functions. Luau does not automatically decide to JIT based on tracing or runtime statistics; compilation is selected explicitly. The native path can use type annotations to specialize code, and it is not available on every platform, so idiomatic code must still perform well in the interpreter. For bytecode-level optimizations such as inlining and loop unrolling, the CLI also supports optimization level 2 (\`-O2\`).`,
    source: `${PERFORMANCE}#function-inlining-and-loop-unrolling`,
    tags: ["performance", "native", "jit"],
    keywords: ["native code", "@native", "--!native", "JIT", "x64", "arm64"],
  },
  {
    id: "fastcalls-and-imports",
    questions: [
      "What is a fastcall in Luau?",
      "How do I make builtin calls fast?",
      "Why localize math.max?",
      "What are global imports in Luau?",
      "Does string.byte or s:byte perform better?",
      "What is namecall?",
      "Why avoid getfenv for performance?",
    ],
    answer: `Luau optimizes builtin calls with a fastcall mechanism. The call must be obvious to the compiler: a direct builtin call such as \`math.max(x, 1)\`, or a localized function such as \`local max = math.max; max(x, 1)\`. Indirect calls and method calls do not get the fast path, so \`string.byte(s)\` is faster than \`s:byte()\`.

Global access chains such as \`math.max\` are also resolved at load time through an optimization called imports. This relies on a predictable environment. \`getfenv\`, \`setfenv\`, and \`loadstring\` can inspect or change that environment, so using any of them marks it impure and disables the optimization. Even reading with \`getfenv\` deoptimizes the environment.

Method calls get a different optimization called namecall, which is why \`obj:Method()\` is preferred over \`obj.Method(obj)\`. For namecall to work well, \`__index\` should point directly to a table, ideally the table itself, rather than to a function or a deep chain.`,
    source: `${PERFORMANCE}#specialized-builtin-function-calls`,
    tags: ["performance", "builtins"],
    keywords: ["fastcall", "imports", "namecall", "localize", "math.max", "string.byte"],
  },
  {
    id: "table-performance",
    questions: [
      "How do I make Luau table access fast?",
      "What is the fastest way to iterate a table in Luau?",
      "Is table length constant time in Luau?",
      "How do I preallocate a table in Luau?",
      "Why is my table field access slow with metatables?",
      "Should I use pairs or generalized iteration?",
      "How do I build a table of squares efficiently?",
      "What table shape is fastest?",
    ],
    answer: `Keep table access predictable:

- Use field names known at compile time, preferably \`table.field\` or a constant string key.
- Avoid metatables for hot field access. Store data directly in the table and methods in the metatable.
- Keep object shapes uniform. Reusing the same keys and field order helps inline caching.

For iteration, generalized iteration, \`pairs\`, and \`ipairs\` are comparable because Luau emits specialized bytecode for the common forms. Generalized iteration is recommended when you do not need \`ipairs\` semantics. \`for i = 1, #t\` is usually slightly slower because of the extra table reads.

\`#t\` is guaranteed to find the array boundary in the array part, and its worst-case complexity is O(logN), often effectively constant because the length is cached after \`table.insert\` and \`table.remove\`.

For creation, use a literal with all fields when the shape is known. For arrays, \`table.create\` preallocates capacity, and sequential indices are faster than repeated \`table.insert\` when the final size is known:

\`\`\`luau
local squares = table.create(100)
for i = 1, 100 do
    squares[i] = i * i
end
\`\`\`

An ideal object is a table whose metatable's \`__index\` points to the table itself. Avoid \`__index\` functions and deep chains in hot code.`,
    source: `${PERFORMANCE}#inline-caching-for-table-and-global-access`,
    tags: ["performance", "tables"],
    keywords: ["table access", "inline caching", "table.create", "generalized iteration", "table length", "__index"],
  },
  {
    id: "fenv-loadstring",
    questions: [
      "Why should I avoid getfenv in Luau?",
      "Why is loadstring slow?",
      "What does setfenv do to optimizations?",
      "Does loadstring work in Luau?",
      "Why are globals slower than locals?",
      "What is a pure environment?",
      "How do I avoid dynamic deoptimization?",
    ],
    answer: `Luau still supports \`getfenv\`, \`setfenv\`, and \`loadstring\` for backwards compatibility, but they are performance hazards. They can inspect or modify a function's environment, so the VM must treat that environment as impure and disable optimizations such as global imports and inline caching.

\`loadstring\` can also load additional code that runs in the caller's environment, which has the same effect. Reading with \`getfenv\` is enough to deoptimize the environment, even if the code never writes to it.

Prefer locals and \`require\`. Localize hot globals yourself, or rely on the import optimization, which only applies to pure environments. Avoid dynamic environment manipulation unless you actually need it.`,
    source: `${PERFORMANCE}#importing-global-access-chains`,
    tags: ["performance", "environment"],
    keywords: ["getfenv", "setfenv", "loadstring", "pure environment", "deoptimization"],
  },
  {
    id: "gc-allocation",
    questions: [
      "How does garbage collection work in Luau?",
      "How do I reduce GC pauses in Luau?",
      "What is an incremental garbage collector?",
      "Should I avoid allocations in loops?",
      "What are GC assists?",
      "What is a shrinkable weak table?",
      "Does Luau have a generational GC?",
    ],
    answer: `Luau uses an incremental garbage collector with three phases: mark, atomic, and sweep. It does not stop the world to traverse the whole heap. Instead, the collector does a little work as the program allocates, through "GC assists".

For high-performance code, avoid allocating temporary tables and userdata in tight loops. The allocator is fast, but every allocation adds GC work and can interrupt throughput. In an engine, you can also request garbage collection once per frame so most of the work happens outside script execution.

Large or frequently modified tables can make the atomic phase expensive. Weak tables can opt into shrinking by including \`s\` in their \`__mode\` string, which resizes them during collection; this can cause missing keys if the table is iterated while it shrinks.

Luau does not currently implement Lua 5.4's generational GC mode. It focuses on incremental collection, paged sweeping, and reduced pauses.`,
    source: `${PERFORMANCE}#improved-garbage-collector-pacing`,
    tags: ["performance", "garbage collection"],
    keywords: ["garbage collector", "incremental", "GC assist", "allocation", "weak table"],
  },
  {
    id: "linter-overview",
    questions: [
      "What is the Luau linter?",
      "How is the linter different from the type checker?",
      "How do I disable a lint?",
      "How do I disable all lints?",
      "What is luau-analyze?",
      "Does the linter run in strict mode?",
      "What are Luau lint warnings?",
    ],
    answer: `The Luau linter is separate from the type checker. It produces opinionated warnings about code that is likely to be wrong or confusing, while the type checker models types and reports mismatches. \`luau-analyze\` runs both.

Most lint warnings are enabled by default. Disable one at the top of a file with \`--!nolint NAME\`, or disable all of them with \`--!nolint\`. Disabling lints does not disable type checking; that requires \`--!nocheck\`. Conversely, \`--!nocheck\` does not silence the linter.

The in-browser Luau playground on the official site does not currently run the linter. Use the CLI or the editor integration to see lint warnings.`,
    source: `${LINT}#lint-warnings`,
    tags: ["lint", "tooling"],
    keywords: ["linter", "luau-analyze", "--!nolint", "warning", "type checker"],
  },
  {
    id: "lint-list",
    questions: [
      "What does UnknownGlobal mean?",
      "What is LocalShadow?",
      "What is LocalUnused?",
      "What does ForRange warn about?",
      "What is MisleadingAndOr?",
      "What is UnreachableCode?",
      "What does TableOperations warn about?",
      "What is ComparisonPrecedence?",
    ],
    answer: `Luau's linter has these warnings (name and numeric code):

- UnknownGlobal (1): a global that is not builtin and not defined in the script. In strict or nonstrict mode the type checker catches this instead.
- DeprecatedGlobal (2): a compatibility global whose use is discouraged.
- GlobalUsedAsLocal (3): a global used in only one function that could be a local.
- LocalShadow (4): a local shadows another local or a global.
- SameLineStatement (5): several statements on one line.
- MultiLineStatement (6): one statement spread over badly indented lines.
- LocalUnused (7): an unused local. Prefix with \`_\` to silence it.
- FunctionUnused (8): an unused function.
- ImportUnused (9): an unused \`require\` result.
- BuiltinGlobalWrite (10): writing to a builtin global such as \`table\`.
- PlaceholderRead (11): reading the \`_\` placeholder.
- UnreachableCode (12): code that cannot run.
- UnknownType (13): an invalid type name in a string comparison or API call.
- ForRange (14): a numeric for that runs zero or one times, or has a suspicious step.
- UnbalancedAssignment (15): a mismatch between the number of variables and values.
- ImplicitReturn (16): a function sometimes returns a value and sometimes implicitly returns nothing.
- DuplicateLocal (17): duplicate parameter or local names.
- FormatString (18): an invalid literal format string passed to a library function.
- TableLiteral (19): duplicate or ignored keys in a table literal.
- UninitializedLocal (20): a local used before being assigned.
- DuplicateFunction (21): two functions with the same name in one scope.
- DeprecatedApi (22): use of a deprecated API.
- TableOperations (23): incorrect or inefficient array operations.
- DuplicateCondition (24): the same condition checked twice.
- MisleadingAndOr (25): \`a and b or c\` where \`b\` may be false or nil.
- CommentDirective (26): an unknown or ignored comment directive.
- IntegerParsing (27): an integer literal that is truncated or loses precision.
- ComparisonPrecedence (28): \`not x == y\` or chained comparisons without parentheses.`,
    source: `${LINT}#lint-warnings`,
    tags: ["lint", "warnings"],
    keywords: ["UnknownGlobal", "LocalShadow", "LocalUnused", "ForRange", "MisleadingAndOr", "lint codes"],
  },
  {
    id: "lua-version-support",
    questions: [
      "Is Luau compatible with Lua 5.2?",
      "Does Luau support goto?",
      "Does Luau have 64-bit integers?",
      "Does Luau support bitwise operators?",
      "Does Luau have const variables?",
      "What Lua 5.4 features are in Luau?",
      "Does Luau have string.pack?",
      "Does Luau support __gc?",
    ],
    answer: `Luau is based on Lua 5.1 and adopts selected features from later versions.

Supported from Lua 5.2 and later include yieldable \`pcall\`/\`xpcall\`, \`__len\` on tables, hexadecimal and \`\\z\`/\`\\u\` string escapes, the \`bit32\` library, \`utf8\`, \`string.pack\`/\`unpack\`/\`packsize\`, floor division, \`table.move\`, \`coroutine.isyieldable\`, \`coroutine.close\`, and Lua 5.4-style \`math.random\` based on PCG.

Luau also has \`const\` bindings through its own syntax, even though Lua 5.4's \`<const>\` attribute is not supported.

Not supported: \`goto\`, 64-bit integers, bitwise operators, \`__gc\` finalizers, ephemeron tables, \`__pairs\`/\`__ipairs\`, yielding metamethods, to-be-closed variables, and Lua 5.5 global declarations. Integers and bitwise operators are omitted deliberately: Luau keeps one double-precision number type and uses \`bit32\` instead.`,
    source: `${COMPATIBILITY}#lua-52`,
    tags: ["compatibility", "lua"],
    keywords: ["Lua 5.2", "Lua 5.3", "Lua 5.4", "goto", "integers", "bitwise", "const"],
  },
  {
    id: "lua-differences",
    questions: [
      "How does Luau differ from Lua?",
      "Does Luau support tail calls?",
      "What is different about table assignment order in Luau?",
      "Does Luau call __eq for rawequal objects?",
      "How does os.time differ from Lua?",
      "Are closure identities stable in Luau?",
    ],
    answer: `Luau has a few deliberate behavior differences from Lua 5.x:

- Tail calls are not supported. This simplifies the VM, keeps stack traces predictable, and lets the runtime validate caller identity for security.
- In mixed table literals, assignment follows program order. Lua 5.x sometimes assigns array elements first.
- Equality calls the \`__eq\` metamethod even when the objects are rawequal. This matches other metamethods and helps with NaN checks.
- A \`function()\` expression may reuse a previously created closure when the captured upvalues are identical. This changes object identity but not calling semantics.
- \`os.time\` with a table argument returns a UTC timestamp for consistency.
- \`print\` calls \`__tostring\`, matching Lua 5.4 rather than older Lua behavior.

The comment about closures means you should not rely on two evaluations of the same function expression producing different table keys.`,
    source: `${COMPATIBILITY}#differences-from-lua`,
    tags: ["compatibility", "semantics"],
    keywords: ["tail calls", "table assignment", "__eq", "closure", "os.time", "differences"],
  },
  {
    id: "implementation-limits",
    questions: [
      "What are the limits of a Luau function?",
      "How many locals can a Luau function have?",
      "How many registers does Luau have?",
      "What is the Luau stack depth limit?",
      "How many upvalues can a function have?",
      "How many constants can a Luau function have?",
      "How many nested functions are allowed?",
    ],
    answer: `Luau documents these per-function limits:

- Local variables: 200, including arguments.
- Upvalues: 200.
- Registers: 255, including locals and arguments.
- Constants: 2^23.
- Instructions: 2^23; the limit applies to control flow.
- Nested functions: 2^15.
- Stack depth: 20,000 Lua calls per Lua thread, and 200 C calls per C thread. \`pcall\` and \`coroutine.resume\` nesting counts against the C limit.

These numbers can change as code generation evolves, and code written close to a limit is fragile. They are documented for diagnostics, not as targets. Lua 5.3 has larger upvalue and constant limits, but Luau's limits are sufficient for normal use.`,
    source: `${COMPATIBILITY}#implementation-limits`,
    tags: ["compatibility", "limits"],
    keywords: ["locals", "upvalues", "registers", "constants", "stack depth", "nested functions"],
  },
  {
    id: "sandboxing",
    questions: [
      "Is Luau sandboxed?",
      "What can Luau scripts access?",
      "Why are io and package removed?",
      "Can Luau read files?",
      "How do embedders sandbox Luau?",
      "Can Luau execute shell commands?",
      "What is the Luau sandboxing model?",
    ],
    answer: `Luau is designed to be embedded and sandboxed. By default, a script cannot read files, open sockets, or execute shell commands. The standard library is deliberately minimal, and the host application decides what extra APIs to expose.

Removed for sandboxing include the \`io\` and \`package\` libraries, \`loadfile\`/\`dofile\`, bytecode loading with \`loadstring\`, \`string.dump\`, and dangerous \`os\` functions such as \`os.execute\`. \`getfenv\`/\`setfenv\` remain for compatibility but should be used carefully.

Embedders add their own APIs on top, usually with tagged userdata and restricted access. That is how Roblox and other hosts expose engine functionality while keeping untrusted code contained. The C API also adds coroutine-friendly yields and code coverage support for hosts.`,
    source: `${SANDBOX}#sandboxing`,
    tags: ["sandbox", "security"],
    keywords: ["sandbox", "security", "embedding", "untrusted code", "host API"],
  },
];

export const LUAU_QA_SUGGESTIONS: readonly string[] = [
  "What is Luau?",
  "What are --!strict and --!nonstrict?",
  "How do I iterate over a table in Luau?",
  "What is the difference between any and unknown in Luau?",
  "How do I make a table read-only in Luau?",
  "How do I split a string in Luau?",
  "What does @native do?",
  "Does Luau have the io library?",
];
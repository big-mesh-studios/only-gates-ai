# Luau quick answers

## How do you do a for loop in Luau?

Use a numeric `for` loop when you know how many times to repeat or you have a numeric range:

```luau
for i = 1, 5 do
    print(i)
end
```

The loop variable is local to the loop. Add a third value to change the step:

```luau
for i = 10, 1, -1 do
    print(i)
end
```

To loop over a table, use a generic `for ... in` loop:

```luau
local names = { "Ada", "Grace", "Alan" }

for index, name in names do
    print(index, name)
end
```

`pairs(t)` and `ipairs(t)` also work, but Luau prefers generalized iteration with `for ... in t`.

## How do I write a for loop in Luau?

The numeric form is `for variable = start, finish, step do ... end`. The generic form is `for key, value in table do ... end`.

```luau
for i = 1, 3 do
    print(i)
end

local scores = { 10, 20, 30 }
for index, score in scores do
    print(index, score)
end
```

## How do you write a loop in Luau?

Use a numeric `for` loop for a range, a generic `for ... in` loop for a table, a `while` loop when the condition controls repetition, or `repeat ... until` when the body must run at least once.

```luau
for i = 1, 5 do
    print(i)
end

local items = { "a", "b" }
for index, item in items do
    print(index, item)
end

local count = 0
while count < 3 do
    count += 1
end
```

## How do you loop in Luau?

Use `for` to iterate over a range or table, and `while` or `repeat` for condition-driven loops.

```luau
for i = 1, 3 do
    print(i)
end

local running = true
while running do
    running = false
end
```

## How do I make a for loop?

A numeric for loop is `for i = start, finish, step do ... end`. A generic for loop is `for key, value in table do ... end`.

```luau
for i = 1, 3 do
    print(i)
end
```

## How do you do a for loop?

Use `for` with a start, finish, and optional step:

```luau
for i = 1, 5 do
    print(i)
end
```

## How do you do a while loop in Luau?

A `while` loop repeats while its condition is true:

```luau
local count = 0

while count < 3 do
    print(count)
    count += 1
end
```

Use `break` to leave the loop early:

```luau
while true do
    break
end
```

## How do you do a repeat loop in Luau?

A `repeat ... until` loop runs the body first and checks the condition at the end:

```luau
local count = 0

repeat
    print(count)
    count += 1
until count >= 3
```

The body always runs at least once, even when the condition starts out true.

## How do you break out of a loop in Luau?

Use the `break` statement. It exits the innermost loop:

```luau
for i = 1, 10 do
    if i == 4 then
        break
    end
    print(i)
end
```

## How do you continue a loop in Luau?

Use the `continue` statement. It skips the rest of the current iteration and starts the next one:

```luau
for i = 1, 5 do
    if i % 2 == 0 then
        continue
    end
    print(i)
end
```

`continue` must be the last statement in its block.

## How do you declare a variable in Luau?

Use `local` for a variable scoped to the current block:

```luau
local name = "Ada"
local count = 0
local isReady = true
```

You can also add a type annotation:

```luau
local name: string = "Ada"
local count: number = 0
```

Avoid creating globals. `local` is the normal choice in Luau.

## How do you define a function in Luau?

Use `function` or `local function`:

```luau
local function add(a: number, b: number): number
    return a + b
end

print(add(2, 3))
```

Use `local function` when the function only needs to be visible in the current file or block.

## How do I call a function in Luau?

Call a function by name with parentheses and arguments:

```luau
local function greet(name: string)
    print("Hello, " .. name)
end

greet("Ada")
```

Methods use a colon: `object:method()`.

## How do I add two numbers in Luau?

Use the `+` operator:

```luau
local sum = 2 + 3
print(sum)
```

Luau numbers are doubles by default. Compound assignment also works: `sum += 1`.

## How do I print something in Luau?

Use `print`:

```luau
print("Hello, world!")
print(2 + 3)
```

Use `warn` for warning output and `error` to raise an error.

## How do I check if something is true in Luau?

Use an `if` statement and comparison operators:

```luau
local score = 80

if score >= 80 and score < 90 then
    print("B")
else
    print("other")
end
```

Only `false` and `nil` are falsey. Zero and the empty string are truthy in Luau.

## How do you create a table in Luau?

Use braces. Tables can be arrays, dictionaries, or both:

```luau
local list = { "a", "b", "c" }
local config = { name = "Ada", age = 36 }
local mixed = { "first", 42, enabled = true }
```

Read and write entries with `.` or `[]`:

```luau
print(config.name)
config.age = 37
list[1] = "A"
```

## How do you iterate over a table in Luau?

Use generalized iteration:

```luau
local items = { "sword", "shield", "potion" }

for index, item in items do
    print(index, item)
end
```

For key/value pairs, use the same form:

```luau
local player = { name = "Ada", score = 10 }

for key, value in player do
    print(key, value)
end
```

## How do you write an if statement in Luau?

Use `if`, `elseif`, `else`, and `end`:

```luau
local score = 80

if score >= 90 then
    print("A")
elseif score >= 80 then
    print("B")
else
    print("C")
end
```

`then` and `end` are required. Parentheses around the condition are optional.

## How do you add type annotations in Luau?

Put `: Type` after a variable, parameter, or return type:

```luau
local name: string = "Ada"

local function greet(who: string): string
    return "Hello, " .. who
end
```

Luau checks annotations when the script uses `--!strict` or `--!nonstrict`. You can also define type aliases with `type`.

## How do I write a string in Luau?

Use single or double quotes:

```luau
local name = "Ada"
local greeting = 'Hello'
```

Long strings can use double brackets:

```luau
local text = [[This can span
multiple lines.]]
```

## How do you concatenate strings in Luau?

Use the `..` operator:

```luau
local first = "Ada"
local last = "Lovelace"
local full = first .. " " .. last
print(full)
```

Use `string.format` or string interpolation when you need more control:

```luau
local message = `Hello, {first}!`
print(message)
```

## How do you write a comment in Luau?

Use `--` for a single-line comment and `--[[ ... ]]` for a block comment:

```luau
-- This is a comment
print("Hello") -- This is also a comment

--[[
    This whole block is ignored.
]]
```

Comments that start with `--!` are file directives such as `--!strict` or `--!nonstrict`.

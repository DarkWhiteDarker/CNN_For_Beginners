# The Full Story: CNN Interpretability Project

This document explains the entire project from start to finish, in plain
language, as if telling the story to another beginner AI engineer who
wasn't there while it was built. If you only read one file to understand
this project completely, read this one.

---

## Part 1: The motivation — why this project exists

Most beginner CNN projects follow the same pattern: pick a dataset, build
a model, train it, report an accuracy number, done. That teaches you how
to *use* a CNN, but not what's actually happening inside one.

This project set out to answer a different, harder question: **can I open
up a trained neural network and actually explain, with evidence, what it
learned and why it makes the decisions it does?**

That single goal shaped every decision that followed — the dataset chosen,
the architecture designed, and even which experiments were considered
worth running.

---

## Part 2: The dataset and the philosophy

**Fashion-MNIST** was chosen deliberately: 70,000 small (28×28), grayscale
images of clothing, across 10 categories. It was picked specifically
*because* it's simple — small enough that a person can look at a learned
filter or a feature map and actually reason about it, unlike a large,
high-resolution dataset where "just look at it" stops being practical.

This led to the project's central design rule, repeated throughout:
**every architecture choice is justified by "does this make the network
easier to see inside," not "does this get the best possible accuracy."**
That rule explains almost every unusual-looking decision later in the
story — small filter counts, a shallow network, and a specific pooling
layer choice that exists purely to support a later visualization
technique.

---

## Part 3: Building and training the baseline model

**Files: `src/data_prep.py`, `models/01_baseline_cnn.py`, `src/evaluate.py`**

The data pipeline (`data_prep.py`) does three unglamorous but essential
things: loads the images, rescales pixel values from 0–255 down to 0.0–1.0
(neural networks train better on small, consistent numbers), and adds a
"channel" dimension that Keras's `Conv2D` layers require even for
grayscale images.

The model itself (`01_baseline_cnn.py`) is intentionally small: three
convolutional blocks with only 8, 16, and 32 filters respectively — a
number small enough that every single filter could later be visualized at
once without becoming overwhelming. One specific choice stands out:
instead of the common `Flatten()` layer before the final output,
`GlobalAveragePooling2D()` was used instead. This wasn't about avoiding
a "bug" — it was chosen because that exact architecture pattern
(pooling straight into the output layer) is what makes a later technique,
Grad-CAM, work cleanly.

After training, the model reached **88.83% test accuracy**. That number
alone would be a perfectly good place to stop for a typical project. This
project instead used the confusion matrix as a launching point:
**"Shirt" was the model's clear weak point**, frequently confused with
T-shirt/top and Coat — visually similar categories at this resolution.

That single weak spot became the thread the rest of the project pulled on.

---

## Part 4: Looking inside — three ways of seeing

**File: `src/visualize.py`, run from `notebooks/02_interpretability.ipynb`**

Three different interpretability techniques were built, each answering a
different question:

**Filter visualization** answers: *"what pattern is this specific part of
the network looking for?"* The first layer's 8 filters, displayed as tiny
images, showed recognizable edge- and gradient-detecting patterns.
Deeper layers' filters were shown as grids (rows of filters, columns of
input channels) to make a genuinely important point visible: a deep
layer's filter isn't one simple pattern — it's a combination across every
channel the previous layer produced. This is the literal mechanism of how
hidden layers "talk to" each other.

**Feature map visualization** answers a different question: *"when I feed
in one real image, what does each layer actually produce?"* A Shirt image
was pushed through the network, and its activations were displayed layer
by layer — showing a clear progression from "still looks like the shirt's
outline" (Layer 1) to "abstract, blocky patterns" (Layer 3). This is the
model building increasingly abstract concepts, one layer at a time.

**Grad-CAM** answers the most practically useful question of all:
*"which exact pixels drove this specific decision?"* By computing how
much the last convolutional layer's outputs influenced the model's final
prediction, a heatmap could be generated and overlaid on any image,
highlighting where the network was "looking."

---

## Part 5: The detective work — turning a weakness into a hypothesis

This is the heart of the project's story.

Grad-CAM was run on a correctly-classified Shirt and on several
misclassified ones. A pattern emerged: the model's attention consistently
concentrated on the **collar and upper-shoulder region**. On the correctly
classified example, that region had a distinctive V-neck shape. On the
misclassified examples, the collar was less distinctive — boxier, more
heavily patterned — and the model's attention on that same region led it
toward the wrong answer instead.

This produced a specific, testable hypothesis: **the model relies on
coarse collar/silhouette shape rather than fine details (like buttons),
and this coarse shortcut breaks down on ambiguous examples.**

A hypothesis isn't worth much until it's tested — so two competing fixes
were built and compared.

---

## Part 6: Two experiments, one clear winner

**Files: `models/02_delayed_pooling_cnn.py`, `models/03_class_weighted_cnn.py`**

**Experiment A — more capacity.** An extra convolutional layer was added
before the first pooling step, on the theory that preserving fine detail
longer might let the network notice something like buttons. Result:
Shirt's F1 score slightly *worsened* (0.69 → 0.67). The fix didn't work.

**Experiment B — a stronger training incentive.** The exact original
architecture was kept unchanged, but Shirt examples were weighted 3x more
heavily during training — directly telling the model "mistakes on this
class cost more." Result: Shirt's F1 improved (0.69 → 0.71), recall
jumped from 0.66 to 0.75, and overall test accuracy even improved
slightly too.

**The conclusion connects both results into one clear lesson:** the
model's weakness wasn't caused by a lack of *capacity* to learn a better
feature — it had that capacity all along. It was caused by nothing in
training pushing it hard enough to prioritize learning one. Changing the
incentive fixed it; adding more room to learn, alone, did not. This is a
genuinely important, slightly counterintuitive lesson in how neural
networks actually learn.

---

## Part 7: Proving the foundations — no more black boxes

**Notebook: `notebooks/03_numpy_from_scratch.ipynb`**

Every technique up to this point used Keras's `Conv2D` as a trusted but
unopened box. The final piece of the project opened that box completely:
the convolution operation was implemented from scratch, using only NumPy
and plain loops — sliding a window across an image, multiplying,
summing, adding a bias, exactly as the mathematics describes.

The real, trained weights from the baseline model were then extracted and
fed into this hand-written function, and its output was compared directly
against Keras's own real output for the same layer, same image. They
matched, down to tiny floating-point rounding differences. This is proof,
not just explanation, that the mechanics of a `Conv2D` layer are fully
understood.

---

## Part 8: How the pieces connect — the full sequence

For a beginner following this project end to end, here is the order
everything happens in, and why each step needed the one before it:

1. **Data must be loaded and shaped correctly** before anything else is
   possible (Step 2).
2. **A model architecture must exist** before it can be trained (Step 4).
3. **Training produces a result to investigate** — without a trained
   model, there's nothing to visualize (Step 5).
4. **The trained model's weakness (Shirt) is discovered through its own
   results** — the confusion matrix is what makes Steps 6–8 worth doing
   at all, rather than being a generic exercise.
5. **Filters and feature maps build the vocabulary** needed to understand
   Grad-CAM's output — you can't interpret "the model is looking at the
   collar" meaningfully without first understanding what a feature map
   and a filter even are (Steps 6–7 before Step 8).
6. **Grad-CAM turns the vague weakness into a specific, testable
   hypothesis** — this is the pivot point of the whole project (Step 8).
7. **The hypothesis is tested with controlled experiments**, changing one
   variable at a time so the result is actually interpretable
   (Experiments A and B).
8. **The framework's internals are proven correct independently**, so the
   whole project rests on demonstrated understanding, not blind trust in
   a library (Step 9).
9. **Everything is written up and connected**, so the story is
   reconstructable by someone who wasn't there (Step 10 — the README, and
   this document).

---

## Part 9: What to take away from this project, in one paragraph

A convolutional neural network is not a mysterious black box — its
learned filters can be displayed as images, its intermediate activations
can be extracted and plotted, and the exact pixels driving any single
decision can be highlighted and inspected. When a model has a specific
weakness, that weakness can be investigated systematically: form a
hypothesis from what you observe, test it with a controlled experiment,
and interpret the result honestly — even when, especially when, the
result isn't what you expected. And underneath every framework call,
there is real, checkable mathematics — worth verifying for yourself at
least once, so every future use of a library layer rests on genuine
understanding rather than trust alone.
## Visual CoT for VLA (BYOVLA)
We refer to BYOVLA to implement a visual CoT that excludes the irrelvant elements before VLA conduct the thinking and expose action policy.

Here, the task-irrelevant regions are extracted as follows.

```bash
python byovla.py
```

And it returns

```log
init vlm output:
{"not_relevant_objects": ["hot dog", "strawberry", "broccoli", "colander"], "not_relevant_backgrounds": ["white tile"]}

after refinement, not relevant object:
['hot dog', 'strawberry', 'broccoli', 'colander']not relevant background:
['white tile']
```

It shows that with an outlet VLM we can conduct the precompute visual CoT to eliminate irrelevant visual elements, which may interfere with the VLA action policy. By doing so it improves the reasoning capability of the VLA model.
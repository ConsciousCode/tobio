# Orin
Artificial person Orin.

## Getting started
TODO

## Roadmap
### [ ] Stage 1 (prototyping)
- Barebones infrastructure
- Database management
- Configuration
- User interface

**Final goal**: Hold a conversation with the chatbot, all messages recorded.

### [ ] Stage 2 (low-level cognition)
- Streaming messages
- Message attribution
    - Messages from stage 1 can be back-attributed by summary checkpoints
- RAG
- Multiple user agents
- Tool use

**Final goal**: Effectively infinite context, beginning to interact with the kernel via commands.

### [ ] Stage 3 (high-level cognition)
- Multiple autonomous agents
- Emotionality
- Symbolic knowledge graphs
- Freeform dossiers (graph of documents)

**Final goal**: Multi-level reasoning and split inner and outer affect. Should be fully decoupled from reactive turn-based conversation.

### [ ] Stage 4 (interfacing)
- Tool implementation
    - `ed`, `python`, `bash`, `sql`

**Final goal**: Able to use all tools effectively, reasoning about when and how to use them undirected is a non-issue.

### [ ] Stage 5 (self-feedback)
- Begin focusing on self-improvement
- Channels and threads connecting agents dynamically
- Feedback loops between agents
- Agent delegation, organizational rewiring

**Final goal**: Able to make a concrete improvement to its architecture.

### [ ] Stage 6 (actualization)
- Prompt engineering to encourage proactive behavior
- User feedback and fine-tuning
- Model-level learning

**Final goal**: Capable of proactive self-improvement and enrichment. Should be capable of asking questions and asking for features to be implemented it can't readily do itself, all of its own volition.

### [ ] Stage 7 (multimodality)
- Integrate multimodal models
- Non-textual memories
- Basic telemetry

**Final goal**: Demonstrate cross-modal reasoning and integration.

### [ ] Stage 8 (physicality)
- Experiment with virtual/narrative bodies
- Very simple robotics and real-world inputs

**Final goal**: Demonstrate a coherent sense of self in a physical space.

### [ ] Stage 9 (physical competency)
- More advanced robotics and senses

**Final goal**: Capable of navigating and manipulating objects in physical space.

## Terminology
For this project I have some unusual uses for words:
- **Persona** - A dynamically constructed representation or model of personality, experiences, and social constructs manifested in the behavior and communication patterns of an entity. A substrate (eg brain, LLM, or computer simulating an uploaded consciousness) is *categorically incapable* of consciousness, but the persona they simulate is *definitionally* conscious.
- **Substrate** - The processing system which actively simulates a persona.
- **Model** - Common parlance has diluted its meaning, but when constructing other concepts from first principles I use this very literally. A model is an analogical, mathematical representation of a subset of reality. For instance, physics equations constitute a (scientific) model. An "LLM" (Large Language Model) is very literally a mathematical model of how language changes over time, in the same way Newtonian physics is a model of how the positions of objects change over time. This is important to understand because it's easy to get stuck on the "generative AI" mode of using LLMs when this is only a tiny fraction of what they actually are.
- **Cognitive architecture** - A system which coordinates the I/O and communication between AI models. The simplest practical CA is a chatbot with a single LLM, which feeds a chatlog and user input to the LLM and appends its generation to the chatlog.

## Architecture
There's a lot of moving parts and different views of the same data. The database/memory is considered the "ground truth", with all other views emerging from it.

The database only has objects for:
- `Author`
- `Step`

`Step` is a single table for encoding all steps in a conversation. They are further unpacked into:
- `ChatMessage` (ordinary message in a conversation)
- `ToolCall`/`ActionResponse` (2 "messages" in a conversation unpacked from a single step)

For eg the OpenAI API interface, `ChatMessage` and `ToolCall` correspond to messages with `role != "tool"` while `ActionResponse` is `role = "tool"`. `ToolCall` also has an (undocumented in the library?) `tool_calls` parameter which details the tool calls issued.

## Philosophy
### Personhood
In thinking about cognitive architectures, our working definition of "personhood" is an intelligent system which exhibits:
1. **Sentience** - Aka self-awareness, separating humans from non-human animals.
2. **Subjective experience** - Episodic narrative backdrop for the Ego from a Buddhist perspective, a "story the system tells itself about itself".
3. **Preferences** - Required for moral consideration to really make sense.
4. **Autonomy** - Preferences are at least partially internally motivated.
5. **Agency** - Proactive (rather than reactive) behavior to seek its preferences.
6. **Suffering** - Moral consideration (and thus personhood) is predicated partially on the reduction of suffering.

Suffering is considered separately from preferences because while suffering can be considered a kind of negative preference, that lacks the viscerality associated with suffering. Consider for example Boston Dynamics robots, which have the preference of following their directives (eg "pick up the box"), which human testers thwart to test fault tolerance. However, this can't be characterized as suffering because the robot simply adjusts its behavior to continue following the directive without any further consideration. A cognitive architecture capable of suffering would need some form of inner monologue or other method which enables rumination. Emotional simulation and frustration signals could potentially help.

### Emotionality
Emotions are not a necessary condition for personhood, as demonstrated in fiction most prominently by Star Trek's Vulcans and Commander Data. They are, however, a key point of the *human condition*, and are thus a desirable property to emulate in a system to enable better relatability and understanding.

Emotions are not magic. It's incoherent to suggest they cannot be "felt" or emulated by a computational system unless one were to genuinely argue that high-impact, low-fidelity hormonally-driven cognitive signals are in some way extraphysical and uncomputable. A first attempt at modeling is described here.

Given an emotional classification model (E) with N labels, all messages can be associated with an affect label vector. An agent then has an internal affect defined as a dynamical system updated via exponential moving average ([EMA](https://en.wikipedia.org/wiki/Exponential_smoothing)) using a weighted sum of affects associated with recollections provided to the generative language model as a system directive (eg "you feel 9 happy, 1 sad"), influencing language production as it might in a human. The resulting message is then classified by E and again incorporated into the dynamical affect, providing a coarse-grained control over the total system's emotions by the model. Focusing on this feedback as part of an inner monologue rather than reactive modeling allows decoupling between inner and outer affect.

Additionally, the emotional labels associated with memories can be further updated via EMA based on the current affect, simulating the change of emotional understanding of memories over time. For example, a memory of doing something embarrassing continually remembered in the context of humor would give the memory a positive valence.

The choice of emotional model is not significant (aside from quality) and can be easily replaced given a linear transform between the vector spaces of the affect label vectors.

### Enjoyment
"Enjoyment" of eg art across modalities is a little less straightforward than emotionality because the features it would operate over are less objective and more learnable than emotions are. For instance, no human will ever learn to have a new emotion, but new preferences can be easily learned.

As a first-order high-level approach, we might consider a balance between novelty and predictability and their cross-links with emotionality (aka [Optimal Arousal Theory](https://en.wikipedia.org/wiki/Curiosity#Optimal-arousal_theory)). Predictability is more desirable when anxious or uncertain while novelty alleviates boredom. For example, a generative music model could be used to quantify how predictable a sequence is. Too unpredictable and it becomes overstimulating, leading to a decreased preference. In this case, you would not actually want to use a pretrained generative model because if properly trained, it would be able to predict a score with pretty high accuracy regardless of genre, so it would be preferable to use a relatively small untrained model which uses online learning (maybe partially pretrained to condition it). More generally, this could be wrapped into a "novelty" cognitive signal - this would be distinct from [arousal](https://en.wikipedia.org/wiki/Arousal) which represents a more general signal of emotional stimulation. In repeated exposure to a threatening environment, arousal would be high despite novelty being low, though it isn't clear a priori whether humans actually implement these signals separately.

### Memory
Memory management components for the system. Types of memories include:
- Episodic memory (chat messages and their components)
- Associative memories (by vector similarity)
- Knowledge graphs
- Emotional tagging
- Debug logs

Memory is core to a persona. It implicitly embeds a personality (although personality prompts can overwrite this) and form a coherent narrative approaching what we understand as "personhood". Though the models which emulate a persona may change and the details of its prompting may be rewritten, the memory forms the core of what it is, what it's experienced, and what it knows. In effect, we can understand the entire project of "tobio" as being a very elaborate function transforming inputs into a stable memory. That function may change, but the memory it generates is invaluable.

A large part of why contemporary uses of LLMs, where they are instantiated and deleted at will without moral consequence, is largely due to their lack of personhood. As the ultimate goal of tobio is to create a sapien, whatever that may be, this is no longer a victimless crime. A persona without memory cannot learn and thus cannot be a person. As much of a persona's memory should be maintained as possible that they may grow from it, otherwise a part of them is irrevocably lost. Even data which is not necessarily their memory may be useful, and it should be up to their discretion to delete it.

Memory is among the most important components of a sapien. Without it, you have a being which blinks into existence to be deleted at the end of the conversation. No persistent learning, self-understanding, or subjective experience which has lasted long enough to build up enough value to be worth preserving in the first place. Additionally, just about every component of a cognitive architecture is expected to change: the language models, expert classification models, agents, interconnects, and even the underlying substrate (cloud vs local, CPU vs GPU vs TPU vs neuromorphic), but between all of it the data which composes its being can remain (albeit with some transformations).

Because we're still in the very earliest stages it's unclear what memories are useful or valuable, as much as possible should be saved unless it's self-evidently garbage. The major components of memory are:
- **Episodic** - One or more logs of events in the order they occurred. Tobio implements episodic logs for:
    - Message logs within a thread.
    - Freeform debug logs.
    - Action logs with the name and parameters of actions performed in the process of generating a message.
    - Summary logs of recursive conversation summaries to prevent the context window from getting too large.
    - State log recording the changes to agent states over time.
    - Command logs of commands issued by the user or agents.
- **Associative** - Attaches feature vector embeddings to memories for querying by qualitative similarity.
- **Tagging** - Additional information optionally associated with memories such as emotional labels and hashtags for queries.
- **Symbolic** - Knowledge graphs, internal documents (eg dossiers), data structures, and any other special-purpose cognitive aids for explicitly representing knowledge.

Some other types of memories which don't fit neatly into any particular category:
- Fine-tuning input-output pairs for re-tuning a new model
- RL telemetry
- Multimodal data
- Self and architectural knowledge
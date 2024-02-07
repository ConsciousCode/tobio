At time of writing, the database isn't actually that complicated and could easily be contained in its own file (and the previous commit had that), but expanding its structure in anticipation of other backends helps to clarify the lessons I've learned. ORM has led to the death of 3 earlier attempts because of a lack of clarity and discipline.

Invariant: Database objects always mirror an actual row, and they cannot be constructed without a corresponding row existing.

Database code has too much impedance mismatch with OOP, so a procedural style with methodless objects is ideal. This minimizes coupling between common interfaces (the application's row objects) and the backend which produces and maps them.

The advantages of parity between application code and database schema you get from ORM frameworks like SQLAlchemy is only relevant in rapid prototyping, where duplicating definitions can create subtle inconsistencies which fail late during runtime. Once a project is stable and you're no longer changing the schema every other commit, this stops being relevant but you've incurred the permanent technical debt of all that abstraction.

A major source of confusion and complication in frameworks like SQLAlchemy is that the models are for two distinct kinds represented with a single type: Unbound data and bound rows. The way it handles this is entirely runtime-dependent, with bound rows decaying into unbound data when you exit the session they were created in, and without any type hinting this can lead to very subtle bugs which only emerge at runtime. The main purpose that I can see for having this is to support sugar methods like `add` which are implicitly single dispatch. Convenient and easier to read, but perhaps the wrong approach to implement it.
## Flowchart

flowchart TD
    A[Initialize environment, Q-table, and Blobs] --> B[For each episode 1 to 25000]
    B --> C[Create new Player, Food, and Enemy Blobs]
    C --> D{For each step up to 200}
    D --> E{Choose action}
    E -->|Exploit| F[Use Q-table with probability 1 - epsilon]
    E -->|Explore| G[Random action with probability epsilon]
    F --> H[Player takes action]
    G --> H
    H --> I[Calculate reward]
    I --> J[Update Q-table]
    J --> K[Render environment if show episode]
    K --> D
    D -->|Step limit reached or\nfood found or enemy hit| L[Record episode reward]
    L --> M[Decay epsilon]
    M --> B
    B -->|All episodes complete| N[Plot rewards]
    N --> O[Save Q-table]

    subgraph Key Components
    P[Learning rate α: 0.1]
    Q[Discount factor γ: 0.95]
    R[Exploration rate ε: starts at 0.9, decays by 0.9998 each episode]
    S[Move penalty: 1]
    T[Enemy penalty: 300]
    U[Food reward: 25]
    end
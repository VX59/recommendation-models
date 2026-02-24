# GraphAMP - (Automated Music Preselector)
##### _"Affective computing is the study and development of systems that can recognize, interpret, process, and simulate human affects"_ - wikipedia.
#### GraphAMP is a response to natural-language based music recommendation models.
Rather than attempting to leverage LLMs or complex neural network architecture, GraphAMP is a statistical model that taps into the awesome power of __stochastic modeling__ and __network science__.

The philosophy of GraphAMP follows the idea that __your emotional state when listening to music is similar to the path a drop of water takes on a river__. This document attempts to outline a novel method of modeling emotional state while listening to music. 

Under the hood, GraphAMP is a __Discrete Markov Process__ which is expressed as a __Directed Network__. Nodes represent songs and edges represent transitions. After reading this, you will have a 100% understanding of how recommendations are done in this service which is practically impossible when using recommendation with neural networks.

Let $A$ be an adjacency matrix of a graph which represents all possible transitions between songs in some music library. $A$ is a weighted and directed graph. Let $A$ be the weighted transition from song $i\rightarrow j$ and where $A_{ij} \in [0,1]$. This weight represents something like a personal preference or emotional response to the transition. Its convenient to refer to it as a continuity score as in if we like the transition $i\rightarrow j$ then $A_{ij}$ should be weighted higher.

Continuity scores are updated periodically. Let $\dfrac{dC(i,j)}{dt}$  be a function which evaluates how some connection will be updated over a single time-step.

A continuous transition is defined as listening to all of song $i$ and then all of song $j$, where in between there may be either __none__ or __some arbitrary number__ of songs skipped. The event $i\rightarrow j$ __preserves__ continuity while $i\rightarrow k \rightarrow ...\rightarrow j$ __restores__ continuity.

Let the __continuous session__ be defined as the sequence of songs played which __preserve__ or __restore__ continuity over the course of a listening session such as $S = (s_0, s_1, s_2, ..., s_n)$. We want to encode this into the graph weights.

Let $i, j$ be any 2 songs in a session and lets say we listen to $i$ before $j$. Now suppose $i,j \in S$ and let $\Delta_{ij}$ be the number of steps through $S$ from $i\rightarrow j$. If there are $n$ songs in a session then $\Delta_{ij} \in [1,n-1]$. Let the change in continuity score of the transition $i\rightarrow j$ per time-step be defined as $C_s(i,j) = e^{-\left(\frac{\Delta_{ij}-1}{\sqrt{n}}\right)}$. This ensures that $C_s(i,j) \in (0,1]$. Influence decays exponentially as distance along the continuous session increases at a speed which depends on the length of the session. Then in general for any $i,j$ in a session
$$
C_{s}(i,j) =
\begin{cases}
e^{-\left(\frac{\Delta_{ij}-1}{\sqrt{n}}\right)} & \text{if } i,j \in S \\
0 & \text{else}
\end{cases}
$$
Consider an arbitrary transition from $i,j \in S$ where there exist an arbitrary number of skipped songs in between $i,j$ such as the event $i\rightarrow k \rightarrow ...\rightarrow j$ . Let $K$ be the set of intermediate songs between $i\rightarrow j$ defined as $K = \{k_0, k_1, k_2, ...\}$. It's necessary that we weaken the intermediate connections within this set which is $O(|K|^2)$ . As $K$ grows larger we should weaken the connections more. For any $k_i,k_j \in \text{some }K$ we can penalize $k_i\rightarrow k_j$  with $e^{-|K|^2/|S|}-1$.

We also want to penalize any connections that breaks continuity that is of the form $i\rightarrow K \rightarrow j$  where $K \neq \emptyset$. This penalization can be weighted by instantaneous engagement as in the percentage of the song $k\in K$ that was listened to before skipping. Let $E(k)$ be the percentage of the song listened to and  $\bar E_K = \sum_{k\in K}{E(k)}$ is the average engagement in $K$. Now we can define a continuity penalty function $\Gamma(K)=(e^{-|K|^2/|S|}-1)(1-\bar E_K)$. In general for any $i,j$ in a session
$$
C_w(i,j) = \begin{cases}
e^{-|K|^2/|S|}-1 & \text{if } i,j \in K\\
\Gamma(K) & \text{if } (i \notin K \land j \in K) \lor (j \notin K \land i \in K)\\
0 & \text{else}
\end{cases}
$$

Finally we need to consider individual song level engagement. For any song $i \in S$ we evaluate its __historical engagement__ . Let $f(i)$ be the total historical plays of song $i$ and let $g(i)$ be the historical number of times the song was finished. Then we can update the bidirectional relationship between songs based on their historical engagement alone defined as the function
$$C_e(i,j) = \left( \dfrac{g(i)}{f(i)}\right) \left( \dfrac{g(j)}{f(j)} \right)

$$  
The entire formula for the update equation edge in the session $S$ is defined as
$$
\dfrac{dC(i,j)}{dt} = \left(C_s(i,j)+C_w(i,j)\right)C_e(i,j)
$$
Now at each time step we do $A'_{ij} = A_{ij} + η\dfrac{dC(i,j)}{dt}$ where $0 < η \ll 1$ is a learning rate parameter. Finally the session graph is projected into the rest of the library and for each mutated row of the graph library $A$ we normalize the updated weights and turn them back into probabilities with
$$
\bar A_i = \frac{\operatorname{max(0,\bar A_i)}}{\sum_{i}{\bar A_i}}
$$

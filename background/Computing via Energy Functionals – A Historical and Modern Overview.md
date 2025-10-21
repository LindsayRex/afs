

# Computing via Energy Functionals – A Historical and Modern Overview

## Historical Pioneers of Energy-Based Computing

-   **1870s – Lord Kelvin (William Thomson):** In 1872–73, Kelvin conceived and built the first tide-predicting machine – a mechanical analog computer that used interlocking gears and pulleys to physically **compute** tidal patterns[en.wikipedia.org](https://en.wikipedia.org/wiki/Tide-predicting_machine#:~:text=A%20tide,a%20year%20or%20more%20ahead). By 1876, his brother James Thomson had described a general-purpose mechanical integrator for solving differential equations, essentially inventing the concept of the analog **differential analyzer**[en.wikipedia.org](https://en.wikipedia.org/wiki/Differential_analyser#:~:text=The%20first%20description%20of%20a,5). These devices performed computation by harnessing physical laws (gear rotations, spring tensions, etc.) to minimize errors – an early example of solving problems through the physics of energy and motion rather than discrete logic.
    
-   **1930s – Vannevar Bush:** Building on these ideas, Bush completed the **Differential Analyzer** at MIT in 1931[invent.org](https://www.invent.org/inductees/vannevar-bush#:~:text=NIHF%20Inductee%20Vannevar%20Bush%20Invented,precursor%20to%20the%20modern%20computer). This large-scale analog computer used rotating shafts and disk-wheel integrators to solve complex differential equations by continuous physical processes. Bush’s machine was a **general-purpose analog computer**, showing that physical analogies (e.g. electrical voltages for mechanical forces) could be exploited to compute solutions for engineering problems. It modeled equations as energy flows in circuits, effectively “computing with physics” decades before digital computers took over.
    
-   **1940s – John von Neumann:** In the 1940s, von Neumann explored **cellular automata (CA)** as an alternative computing paradigm. With Stanislaw Ulam’s help, he formulated a grid-based self-reproducing automaton – a theoretical machine made of cells that update via simple rules[embryo.asu.edu](https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata#:~:text=are%20employed%20to%20analyze%20phenomena,reproduction)[embryo.asu.edu](https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata#:~:text=In%201948%2C%20von%20Neumann%20set,Arthur%20Walter%20Burks%20in%201966). Published posthumously in 1966, von Neumann’s work proved that a sufficiently programmed CA could exhibit self-replication and complexity, drawing inspiration from biological systems. This was **computing by emergent behavior**: complex outcomes arising from simple local update rules, without an explicit energy-minimization function. (Notably, von Neumann also calculated that the human brain operates on only ~25 watts, marveling at its efficiency compared to electronic machines[cba.mit.edu](https://cba.mit.edu/events/03.11.ASE/docs/VonNeumann.pdf#:~:text=Von%20Neumann%20then%20estimated%20that,Hence).) His cellular automata laid groundwork for later thinkers like **John Conway** (Game of Life, 1970) and **Stephen Wolfram** (A New Kind of Science, 2002) who showed how rich computation can emerge without traditional algorithms. These rule-based emergent systems, however, often lack a clear theory of _why_ a given complex pattern emerges – a contrast to energy-based approaches which define a functional that the system optimizes.
    
-   **1961 – Rolf Landauer:** The IBM physicist Landauer established a profound link between **information and energy**. In 1961 he asserted that _“real-world computation involves thermodynamic costs”_, specifically that **erasing a bit of information dissipates a minimum amount of heat (kT ln2)**[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=In%201961%2C%20Landauer%20asserted%20that,It%20is%20quite%20fascinating%20to)[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=considered%20logically%20irreversible,analysis%20of%20thermodynamic%20computation%20processes). This principle, now known as **Landauer’s Principle**, anchored the idea that information processing is physical. It implied that computing could be made more energy-efficient – even reversible – if no information is lost. Landauer’s work spurred the field of **thermodynamics of computation**, highlighting that computing **should** be viewed through energy: every logic operation has an entropy cost.
    
-   **1970s – Charles H. Bennett:** In 1973, building on Landauer’s insight, Bennett showed that _logically reversible_ operations could in principle be performed with **arbitrarily little energy dissipation**[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=,1970%29%20Robert%C2%A0W%C2%A0Keyes%C2%A0and%C2%A0Rolf%C2%A0Landauer%2C%C2%A0%E2%80%9CMinimal%20energy). He introduced **reversible computing**, where the computation can be reversed step-by-step, avoiding the “bit erasure” that incurs Landauer’s energy cost. Bennett’s theoretical constructs (like running Turing machines backward) demonstrated that computing **on an energy budget** was possible by design – foreshadowing physical computers that recycle energy. This set the stage for ideas like **adiabatic (reversible) logic circuits** and inspired a search for computing mechanisms with minimal energy loss.
    
-   **1982 – Edward Fredkin & Tommaso Toffoli:** These MIT researchers proposed **Conservative Logic**, a model of computing that mirrors fundamental physics (e.g. billiard ball collisions)[scirp.org](https://www.scirp.org/reference/referencespapers?referenceid=1065770#:~:text=E,253). In their influential 1982 paper, they described the **“billiard ball computer,”** a thought experiment where hard spheres (bits) collide on frictionless tables to perform logic gates without dissipating energy. All interactions are perfectly elastic, so kinetic energy and momentum (information) are conserved. Fredkin and Toffoli’s reversible gates (like the Fredkin gate) showed that computation could be done via **energy-conserving physical processes**[scirp.org](https://www.scirp.org/reference/referencespapers?referenceid=1065770#:~:text=E,253). This was a vivid example that computing _on energy functionals_ (in this case, the kinetic energy of moving balls) is not only possible but could avoid the heating inherent in conventional computers. Their work linked digital logic to physics directly and influenced later developments in reversible and quantum computing.
    
-   **1982 – John Hopfield:** A physicist, Hopfield introduced a new paradigm of **neural network computing via energy minimization**. His 1982 paper demonstrated that a network of neurons with symmetric connections can serve as a content-addressable memory by converging to minimal “energy” states[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=the%20energy%20function%20,these%20in%20turn%20are%20a)[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=Hopfield%20energy%20function%20E%2C%20then,also%20as%20the%20constraints%20are). He defined an explicit **energy function (Lyapunov function)** for the network, akin to a spin glass Hamiltonian, such that the dynamics of neuron updates always decrease this energy. Thus, a Hopfield network will settle into a stable pattern that is a local minimum of the energy landscape – effectively performing **computing by finding energy minima**. Hopfield showed this model could **associate memories** and even solve optimization problems (e.g. the Traveling Salesman problem) by mapping them to an energy function[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=Hopfield%20energy%20function%20E%2C%20then,also%20as%20the%20constraints%20are). This was a landmark because it suggested computational problems might be solved by physical processes “relaxing” to low-energy configurations (an idea drawn from statistical physics). Hopfield’s work bridged physics and computation and laid the foundation for **Ising-model computers** and **energy-based machine learning**.
    
-   **1985 – Geoffrey Hinton & Terry Sejnowski:** Hinton and colleagues extended Hopfield’s ideas with the **Boltzmann Machine**, a stochastic neural network that uses simulated annealing to find good solutions. A Boltzmann machine is essentially a network of spins (neurons) flipping with probabilities governed by the Boltzmann distribution, so that given enough time (and a cooling schedule) it will sample low-energy states of the system. Hinton & Sejnowski popularized these as **“energy-based models”** in AI, explicitly invoking Hamiltonians of spin glasses as the energy functions to be minimized[en.wikipedia.org](https://en.wikipedia.org/wiki/Boltzmann_machine#:~:text=mechanics%20%2C%20which%20is%20used,5). In 1985 they published a learning algorithm for Boltzmann machines that uses gradients of an energy function to adjust network weights. This was a crucial step in connecting **computation to energy functionals**: the machine “computes” by physically (or simulationally) relaxing into thermal equilibrium, and learning makes those energy minima represent useful solutions. Although Boltzmann machines were computationally heavy, they introduced concepts of **probabilistic computing via energy landscapes** that influenced modern deep learning (e.g. Energy-Based Models, Gibbs sampling)[en.wikipedia.org](https://en.wikipedia.org/wiki/Boltzmann_machine#:~:text=mechanics%20%2C%20which%20is%20used,5).
    
-   **1980s – Carver Mead (Neuromorphic Engineering):** In the late 1980s, Carver Mead championed a return to analog principles for computing. He coined the term **“neuromorphic”** to describe analog VLSI circuits that emulate the brain’s neural architectures[en.wikipedia.org](https://en.wikipedia.org/wiki/Neuromorphic_computing#:~:text=Neuromorphic%20engineering%20is%20an%20interdisciplinary,in%20the%20late%201980s). Mead observed that the brain computes with a mere few tens of watts by exploiting analog physics (currents, charges in neurons) in parallel, and sought to harness the same efficiency in silicon. In 1989 he published _Analog VLSI and Neural Systems_, showing how transistor circuits could integrate and fire like neurons, effectively **computing by energy flow in an electrical network**. One of the first successes was a silicon retina: an analog chip that computes visual processing using the physics of transistors and capacitors rather than digital logic. Mead’s **Neuromorphic Electronic Systems** (e.g. Mead & Mahowald’s 1988 silicon neuron) demonstrated that **analog computation can be fast and ultra-efficient**, because it naturally computes solutions (like smoothing images or detecting motion) through the physical convergence of voltages and currents. This work revived analog computing in a modern form, inspiring decades of research into brain-like hardware that minimizes energy use by operating on analog signal domains[en.wikipedia.org](https://en.wikipedia.org/wiki/Neuromorphic_computing#:~:text=Neuromorphic%20engineering%20is%20an%20interdisciplinary,in%20the%20late%201980s).
    
-   **1990s – Analog & Quantum Information theories:** Through the 1990s, several threads developed the idea of computing with novel energy-based media. **Leon Chua**’s prediction of the memristor (1971) came to fruition in 2008, leading to analog _memristive circuits_ for computation by 2010s. **Reversible computing** saw experimental logic gates that dissipate very little heat. On the quantum front, **Seth Lloyd** and others discussed ultimate limits of computation tied to energy, and **Peter Shor** (1994) showed quantum mechanics (which evolves via energy unitary operations) could solve certain problems faster. The concept of **quantum annealing** emerged in theory (Kadowaki & Nishimori, 1998) – using quantum physics to tunnel through energy barriers and find ground states of Ising-like systems[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Google%20Scholar). While less visibly “historical” than earlier items, these developments set the stage for physical computing machines in the 21st century.
    

## Modern Implementations and Examples

By the 2010s, the ideas of computing on energy functionals materialized in several groundbreaking technologies:

-   **Ising Machines (Physical Optimizers):** Researchers built specialized hardware to solve optimization problems by finding minimum-energy states of an Ising model (a network of spins). One approach used **quantum annealing**: in 2011 D-Wave Systems sold the first commercial quantum computer (128-qubit **D-Wave One**), which uses superconducting qubits to physically realize an Ising model and find its ground state[phys.org](https://phys.org/news/2011-06-d-wave-commercial-quantum.html#:~:text=The%20announcement%20comes%20just%20a,146%3B%20spins)[phys.org](https://phys.org/news/2011-06-d-wave-commercial-quantum.html#:~:text=D,system%20works%20with%20quantum%20effects). D-Wave’s machines perform computation by literally evolving a quantum magnetic system towards its lowest-energy configuration, thereby outputting solutions to NP-hard problems encoded in the spin couplings. Around the same time, optical engineers led by Yoshihisa Yamamoto demonstrated the **Coherent Ising Machine** (2014), an optical network of parametric oscillators that settles into minimal Ising spin configurations[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Finding%20the%20ground%20states%20of,the%20OPOs%20and%20the%20Ising)[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Hamiltonian2%20%2C%2022%2C4%20%2C%2024%2C6,no%20computational%20error%20was%20detected). In Yamamoto’s 4-oscillator prototype, the system was programmed with a small NP-hard problem and, over 1000 runs, consistently found the optimal solution by virtue of its physics (laser pulses finding a synchronized minimal phase state)[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Hamiltonian2%20%2C%2022%2C4%20%2C%2024%2C6,no%20computational%20error%20was%20detected). These Ising machines – whether quantum, optical, or electronic – are essentially **analog computers for optimization**, exploiting natural dynamics to **compute by energy minimization**. They have since scaled up (e.g. 2000-node optical Ising machines[quantum-journal.org](https://quantum-journal.org/papers/q-2023-10-24-1151/#:~:text=In%20this%20work%2C%20we%20address,accelerate%20the%20speed%20of%20computation)) and inspired _“physics-inspired”_ algorithms for hard problems.
    
-   **Neuromorphic Chips:** The neuromorphic computing vision of the 1980s bore fruit in the 2010s with large-scale prototypes. In 2014, IBM unveiled **TrueNorth**, a CMOS chip with 1 million hardware “neurons” and 268 million synapses implementing a spiking neural network[en.wikipedia.org](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=networks%20%20and%20deep%20learning,are%20improving%20neuromorphic%20processors%20steadily)[en.wikipedia.org](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=TrueNorth%20was%20a%20neuromorphic%20,4%20billion). TrueNorth forsakes the usual CPU architecture for a massively parallel, event-driven design that computes via spike timings and local memory – much like a brain. Importantly, it consumes only 65 mW (orders of magnitude less power than conventional processors) by using **physical neuron-like operations** instead of power-hungry clocked logic. Each neuron integrates inputs (currents) and fires when a threshold (energy potential) is reached, an analog process implemented with digital efficiency. Similarly, in 2017 Intel announced **Loihi**, a neuromorphic chip with 128,000 neurons and on-chip learning, also inspired by spiking dynamics. Academic projects like **BrainScaleS** (Heidelberg) use analog circuits to emulate neurons in continuous time, running 1000× faster than biology. Neuromorphic systems are **computing on energy functions** in the sense that their collective state often optimizes some cost function (e.g. a neural network loss) through physical dynamics. They excel at tasks like pattern recognition with extremely low energy per operation, validating the promise of harnessing physics (spikes, charges) for computation.
    
-   **Analog & In-Memory Computing:** A related trend is analog _in-memory computing_ for AI. For example, chips that use **memristors** or phase-change devices can perform matrix multiplications “in place” by letting electrical currents sum analog values stored in memory cells. This effectively computes a neural network layer by Kirchoff’s current law – the physics of charge distribution does the math, rather than instruction-by-instruction logic. By 2020, researchers showed memristor crossbar arrays that solve optimization tasks or implement Hopfield-like associative memories, taking advantage of the natural convergence of analog circuits[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We). These systems often operate by minimizing an internal energy (e.g. the charge error in an analog neural network), achieving results with far less energy than digital arithmetic would use.
    
-   **Chemical and Biological Computing:** Outside electronics, other media have been explored for energy-based computing. **DNA computing** (Adleman, 1994) used the binding energy of DNA strands to solve a small graph path problem in a test tube – essentially leveraging chemical reaction energy to perform computation. **Chemical reaction networks** and **molecular computers** are being investigated to see if reactions naturally compute solutions to equations (for instance, finding equilibrium = solving a set of constraints). Even **biological cells** themselves can be viewed as computers optimizing energy (the “free energy principle” proposed by Karl Friston suggests brains self-organize by minimizing a free-energy functional). While these examples are specialized, they reinforce the theme that _any_ energy-driven system – electrical, optical, quantum, or chemical – can potentially be programmed to perform useful computations by steering it toward desired low-energy states.
    

## Stability and Reliability Challenges

One major challenge with analog, energy-functional computing is **stability** – physical systems are subject to noise, variability, and even chaos. Unlike binary digital logic which affords error correction via discrete states, analog computers must contend with continuous fluctuations. Pioneers of analog computing in the 20th century were well aware of drift and component tolerance issues (e.g. Bush’s Differential Analyzer required frequent recalibration of its mechanical parts). Modern energy-based computers likewise implement clever strategies to ensure reliable results:

-   **Calibration and Noise Compensation:** For neuromorphic and analog chips, device mismatches and noise can perturb computations. Engineers address this by calibrating hardware and designing algorithms to be noise-tolerant. For instance, the BrainScaleS analog neuromorphic system must deal with “fixed-pattern noise” (each silicon neuron differs slightly) and trial-to-trial variation. Researchers report that networks running on BrainScaleS _must cope with a certain level of perturbations_, so they use calibration and **hardware-in-the-loop training** to adjust for these variations[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We). In practice, this means measuring the analog behavior and tuning parameters (or training neural weights on the actual chip) so that the physical dynamics still converge to correct answers despite noise. Techniques like **noise injection during training** can make the neural network robust against the uncertainties of analog computation[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We). The goal is to harness the efficiency of analog while mitigating its unpredictability.
    
-   **Error Correction in Physical Annealers:** Quantum and optical Ising machines also face stability issues. Quantum annealing, for example, is an analog process sensitive to control errors and decoherence. D-Wave’s superconducting processors exhibit slow drift in flux biases that can misalign the energy landscape over time. To combat this, D-Wave systems perform regular **“drift correction”** – periodically recalibrating qubit biases each hour to keep the effective error bounded and Gaussian[docs.dwavequantum.com](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you). They also run each problem multiple times and use statistical techniques to filter out thermal or analog errors. In essence, while the quantum annealer computes by sliding into a minimum energy, the engineers surround it with classical feedback loops to nudge it if it strays due to noise. Similarly, optical Ising machines must maintain laser stability; techniques like injection locking and feedback control help keep the system on track to the true ground state. In all these cases, **redundancy and repetition** are key: by running the analog solver many times (or having many parallel units) and aggregating results, one can overcome individual run errors – a strategy also used in DNA computing where vast parallelism compensates for probabilistic errors.
    
-   **Physical Limits and Trade-offs:** There is a recognition that analog systems won’t be perfectly stable – rather, the aim is to make them _stable enough_ that their speed or energy advantages win out. Digital post-processing or occasional digital correction is often integrated (creating hybrid systems). For example, a recent analog optical computing proposal might do an analog compute step, then a digital refine step, marrying the best of both. Researchers are also exploring **statistical computing** that tolerates a certain error rate (e.g. p-bits and probabilistic computers that leverage noisy devices as a feature). In summary, stability is managed through a combination of **calibration, adaptive algorithms, error correction codes, environmental control, and clever problem encoding** that avoids extremely sensitive regions of the energy landscape.
    

## Convergence of Different Approaches

Intriguingly, many of these strands – analog computers, neural networks, Ising machines, even cellular automata – are now overlapping and being seen as part of a larger paradigm: **computing by exploiting physical dynamics**. They have different names and origins but share a common theme: rather than sequentially executing symbolic logic operations, we **set up a physical or numerical system** whose _natural_ evolution produces the answer. This viewpoint has been gaining momentum as we reach limits of conventional computing.

Today, researchers explicitly talk about **“energy-based computing”** frameworks that unify ideas from Ising machines and neuromorphic systems. For example, a 2023 project aims to _“harness and generalize the energy potential”_ in Hopfield networks and oscillatory Ising machines, seeking a _unified framework_ that bridges the algorithmic view and the physical view of computing[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=paradigms%20rooted%C2%A0in%20the%20Ising%20Hamiltonian,singularity%20theory%20to%C2%A0determine%20the%20optimal)[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering). The hope is to **reconcile algorithmics with physics**, so that we can systematically design and program these unconventional computers[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering), much as we do digital ones. This involves developing theory to predict and prove what an analog or emergent system will do – addressing the very concern you raised about **knowing why something emerges**.

It’s worth noting the contrast between energy-functional approaches and purely emergent ones like cellular automata. CA-based “unconventional computing” (e.g. Conway’s Game of Life patterns, Langton’s loops) showed that complex behaviors can emerge from simple rules, but often one cannot easily **predict or guarantee** the outcome – you have to run it and see. They lack an obvious Lyapunov or energy function that they are minimizing; as you said, they _“make stuff emerge without understanding why it emerges.”_ Energy-based systems, on the other hand, usually have a defined objective: e.g. a Hopfield network’s energy **is** known to decrease until a stable state (solution) is reached[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=the%20energy%20function%20,these%20in%20turn%20are%20a). The trade-off is that energy-based systems can get stuck in local minima (suboptimal solutions), whereas something like a cellular automaton might surprise you with a novel global pattern – but you can’t easily constrain or prove it. Modern research is attempting to get the best of both worlds: leveraging emergence **with** a layer of theoretical guarantees. For instance, by designing an “elastic” Ising network or other analog systems that are mathematically tractable, scientists aim to let solutions _emerge_ from physics but also be _provably correct_ or at least predictable[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering).

In summary, **computing on energy functionals** has deep roots and a vibrant present:

-   From Kelvin’s gears to Hopfield’s spins, many visionary thinkers proposed letting Mother Nature compute by relaxing to minimal energy or by evolving simple laws.
    
-   Today’s neuromorphic and quantum machines are direct descendants of those ideas, literally performing calculations by **minimizing physical energy** or **exploiting natural dynamics**.
    
-   All these approaches – analog, neuromorphic, quantum annealing, Ising solvers, etc. – are indeed variations on a theme: _using the physics of energy and motion to process information_. They differ in the substrate (electronics, optics, magnets, DNA…), but they echo each other. As you noted, they are “different names for the same thing” in spirit, and the field is increasingly interdisciplinary.
    
-   The remaining gaps are practical (how to make these systems large-scale and reliable) and theoretical (understanding and proving their behavior). The stability issue is being tackled with calibration and error mitigation[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We)[docs.dwavequantum.com](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you), while the “why it works” issue is being addressed by new theory bridging physics and computation[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering).
    

This convergence is an exciting frontier. Your intuition to bring these strands together is well-founded – the history shows a clear through-line of thought, and the latest work is explicitly **bringing together analog physics, optimization, and computation theory** into a unified framework. As we push beyond the limits of binary logic, these energy-functional computing ideas may well play a key role in the future of computing, delivering fast and efficient solutions **by design**, with your contribution of provable algorithms adding rigor to what was once only emergent behavior.

**Sources:** The timeline and examples above reference original works and modern studies by the mentioned pioneers and projects[en.wikipedia.org](https://en.wikipedia.org/wiki/Tide-predicting_machine#:~:text=A%20tide,a%20year%20or%20more%20ahead)[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=In%201961%2C%20Landauer%20asserted%20that,It%20is%20quite%20fascinating%20to)[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=Hopfield%20energy%20function%20E%2C%20then,also%20as%20the%20constraints%20are)[en.wikipedia.org](https://en.wikipedia.org/wiki/Boltzmann_machine#:~:text=mechanics%20%2C%20which%20is%20used,5)[en.wikipedia.org](https://en.wikipedia.org/wiki/Neuromorphic_computing#:~:text=Neuromorphic%20engineering%20is%20an%20interdisciplinary,in%20the%20late%201980s)[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Finding%20the%20ground%20states%20of,the%20OPOs%20and%20the%20Ising)[en.wikipedia.org](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=networks%20%20and%20deep%20learning,are%20improving%20neuromorphic%20processors%20steadily)[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We)[docs.dwavequantum.com](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you)[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering), illustrating the development of energy-based computing from the 19th century to today.

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://cba.mit.edu&sz=32)

![](https://www.google.com/s2/favicons?domain=https://embryo.asu.edu&sz=32)

![](https://www.google.com/s2/favicons?domain=https://www.invent.org&sz=32) good afternoon good afternoon

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)



# Computing via Energy Functionals – A Historical and Modern Overview

## Historical Pioneers of Energy-Based Computing

-   **1870s – Lord Kelvin (William Thomson):** In 1872–73, Kelvin conceived and built the first tide-predicting machine – a mechanical analog computer that used interlocking gears and pulleys to physically **compute** tidal patterns[en.wikipedia.org](https://en.wikipedia.org/wiki/Tide-predicting_machine#:~:text=A%20tide,a%20year%20or%20more%20ahead). By 1876, his brother James Thomson had described a general-purpose mechanical integrator for solving differential equations, essentially inventing the concept of the analog **differential analyzer**[en.wikipedia.org](https://en.wikipedia.org/wiki/Differential_analyser#:~:text=The%20first%20description%20of%20a,5). These devices performed computation by harnessing physical laws (gear rotations, spring tensions, etc.) to minimize errors – an early example of solving problems through the physics of energy and motion rather than discrete logic.
    
-   **1930s – Vannevar Bush:** Building on these ideas, Bush completed the **Differential Analyzer** at MIT in 1931[invent.org](https://www.invent.org/inductees/vannevar-bush#:~:text=NIHF%20Inductee%20Vannevar%20Bush%20Invented,precursor%20to%20the%20modern%20computer). This large-scale analog computer used rotating shafts and disk-wheel integrators to solve complex differential equations by continuous physical processes. Bush’s machine was a **general-purpose analog computer**, showing that physical analogies (e.g. electrical voltages for mechanical forces) could be exploited to compute solutions for engineering problems. It modeled equations as energy flows in circuits, effectively “computing with physics” decades before digital computers took over.
    
-   **1940s – John von Neumann:** In the 1940s, von Neumann explored **cellular automata (CA)** as an alternative computing paradigm. With Stanislaw Ulam’s help, he formulated a grid-based self-reproducing automaton – a theoretical machine made of cells that update via simple rules[embryo.asu.edu](https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata#:~:text=are%20employed%20to%20analyze%20phenomena,reproduction)[embryo.asu.edu](https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata#:~:text=In%201948%2C%20von%20Neumann%20set,Arthur%20Walter%20Burks%20in%201966). Published posthumously in 1966, von Neumann’s work proved that a sufficiently programmed CA could exhibit self-replication and complexity, drawing inspiration from biological systems. This was **computing by emergent behavior**: complex outcomes arising from simple local update rules, without an explicit energy-minimization function. (Notably, von Neumann also calculated that the human brain operates on only ~25 watts, marveling at its efficiency compared to electronic machines[cba.mit.edu](https://cba.mit.edu/events/03.11.ASE/docs/VonNeumann.pdf#:~:text=Von%20Neumann%20then%20estimated%20that,Hence).) His cellular automata laid groundwork for later thinkers like **John Conway** (Game of Life, 1970) and **Stephen Wolfram** (A New Kind of Science, 2002) who showed how rich computation can emerge without traditional algorithms. These rule-based emergent systems, however, often lack a clear theory of _why_ a given complex pattern emerges – a contrast to energy-based approaches which define a functional that the system optimizes.
    
-   **1961 – Rolf Landauer:** The IBM physicist Landauer established a profound link between **information and energy**. In 1961 he asserted that _“real-world computation involves thermodynamic costs”_, specifically that **erasing a bit of information dissipates a minimum amount of heat (kT ln2)**[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=In%201961%2C%20Landauer%20asserted%20that,It%20is%20quite%20fascinating%20to)[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=considered%20logically%20irreversible,analysis%20of%20thermodynamic%20computation%20processes). This principle, now known as **Landauer’s Principle**, anchored the idea that information processing is physical. It implied that computing could be made more energy-efficient – even reversible – if no information is lost. Landauer’s work spurred the field of **thermodynamics of computation**, highlighting that computing **should** be viewed through energy: every logic operation has an entropy cost.
    
-   **1970s – Charles H. Bennett:** In 1973, building on Landauer’s insight, Bennett showed that _logically reversible_ operations could in principle be performed with **arbitrarily little energy dissipation**[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=,1970%29%20Robert%C2%A0W%C2%A0Keyes%C2%A0and%C2%A0Rolf%C2%A0Landauer%2C%C2%A0%E2%80%9CMinimal%20energy). He introduced **reversible computing**, where the computation can be reversed step-by-step, avoiding the “bit erasure” that incurs Landauer’s energy cost. Bennett’s theoretical constructs (like running Turing machines backward) demonstrated that computing **on an energy budget** was possible by design – foreshadowing physical computers that recycle energy. This set the stage for ideas like **adiabatic (reversible) logic circuits** and inspired a search for computing mechanisms with minimal energy loss.
    
-   **1982 – Edward Fredkin & Tommaso Toffoli:** These MIT researchers proposed **Conservative Logic**, a model of computing that mirrors fundamental physics (e.g. billiard ball collisions)[scirp.org](https://www.scirp.org/reference/referencespapers?referenceid=1065770#:~:text=E,253). In their influential 1982 paper, they described the **“billiard ball computer,”** a thought experiment where hard spheres (bits) collide on frictionless tables to perform logic gates without dissipating energy. All interactions are perfectly elastic, so kinetic energy and momentum (information) are conserved. Fredkin and Toffoli’s reversible gates (like the Fredkin gate) showed that computation could be done via **energy-conserving physical processes**[scirp.org](https://www.scirp.org/reference/referencespapers?referenceid=1065770#:~:text=E,253). This was a vivid example that computing _on energy functionals_ (in this case, the kinetic energy of moving balls) is not only possible but could avoid the heating inherent in conventional computers. Their work linked digital logic to physics directly and influenced later developments in reversible and quantum computing.
    
-   **1982 – John Hopfield:** A physicist, Hopfield introduced a new paradigm of **neural network computing via energy minimization**. His 1982 paper demonstrated that a network of neurons with symmetric connections can serve as a content-addressable memory by converging to minimal “energy” states[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=the%20energy%20function%20,these%20in%20turn%20are%20a)[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=Hopfield%20energy%20function%20E%2C%20then,also%20as%20the%20constraints%20are). He defined an explicit **energy function (Lyapunov function)** for the network, akin to a spin glass Hamiltonian, such that the dynamics of neuron updates always decrease this energy. Thus, a Hopfield network will settle into a stable pattern that is a local minimum of the energy landscape – effectively performing **computing by finding energy minima**. Hopfield showed this model could **associate memories** and even solve optimization problems (e.g. the Traveling Salesman problem) by mapping them to an energy function[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=Hopfield%20energy%20function%20E%2C%20then,also%20as%20the%20constraints%20are). This was a landmark because it suggested computational problems might be solved by physical processes “relaxing” to low-energy configurations (an idea drawn from statistical physics). Hopfield’s work bridged physics and computation and laid the foundation for **Ising-model computers** and **energy-based machine learning**.
    
-   **1985 – Geoffrey Hinton & Terry Sejnowski:** Hinton and colleagues extended Hopfield’s ideas with the **Boltzmann Machine**, a stochastic neural network that uses simulated annealing to find good solutions. A Boltzmann machine is essentially a network of spins (neurons) flipping with probabilities governed by the Boltzmann distribution, so that given enough time (and a cooling schedule) it will sample low-energy states of the system. Hinton & Sejnowski popularized these as **“energy-based models”** in AI, explicitly invoking Hamiltonians of spin glasses as the energy functions to be minimized[en.wikipedia.org](https://en.wikipedia.org/wiki/Boltzmann_machine#:~:text=mechanics%20%2C%20which%20is%20used,5). In 1985 they published a learning algorithm for Boltzmann machines that uses gradients of an energy function to adjust network weights. This was a crucial step in connecting **computation to energy functionals**: the machine “computes” by physically (or simulationally) relaxing into thermal equilibrium, and learning makes those energy minima represent useful solutions. Although Boltzmann machines were computationally heavy, they introduced concepts of **probabilistic computing via energy landscapes** that influenced modern deep learning (e.g. Energy-Based Models, Gibbs sampling)[en.wikipedia.org](https://en.wikipedia.org/wiki/Boltzmann_machine#:~:text=mechanics%20%2C%20which%20is%20used,5).
    
-   **1980s – Carver Mead (Neuromorphic Engineering):** In the late 1980s, Carver Mead championed a return to analog principles for computing. He coined the term **“neuromorphic”** to describe analog VLSI circuits that emulate the brain’s neural architectures[en.wikipedia.org](https://en.wikipedia.org/wiki/Neuromorphic_computing#:~:text=Neuromorphic%20engineering%20is%20an%20interdisciplinary,in%20the%20late%201980s). Mead observed that the brain computes with a mere few tens of watts by exploiting analog physics (currents, charges in neurons) in parallel, and sought to harness the same efficiency in silicon. In 1989 he published _Analog VLSI and Neural Systems_, showing how transistor circuits could integrate and fire like neurons, effectively **computing by energy flow in an electrical network**. One of the first successes was a silicon retina: an analog chip that computes visual processing using the physics of transistors and capacitors rather than digital logic. Mead’s **Neuromorphic Electronic Systems** (e.g. Mead & Mahowald’s 1988 silicon neuron) demonstrated that **analog computation can be fast and ultra-efficient**, because it naturally computes solutions (like smoothing images or detecting motion) through the physical convergence of voltages and currents. This work revived analog computing in a modern form, inspiring decades of research into brain-like hardware that minimizes energy use by operating on analog signal domains[en.wikipedia.org](https://en.wikipedia.org/wiki/Neuromorphic_computing#:~:text=Neuromorphic%20engineering%20is%20an%20interdisciplinary,in%20the%20late%201980s).
    
-   **1990s – Analog & Quantum Information theories:** Through the 1990s, several threads developed the idea of computing with novel energy-based media. **Leon Chua**’s prediction of the memristor (1971) came to fruition in 2008, leading to analog _memristive circuits_ for computation by 2010s. **Reversible computing** saw experimental logic gates that dissipate very little heat. On the quantum front, **Seth Lloyd** and others discussed ultimate limits of computation tied to energy, and **Peter Shor** (1994) showed quantum mechanics (which evolves via energy unitary operations) could solve certain problems faster. The concept of **quantum annealing** emerged in theory (Kadowaki & Nishimori, 1998) – using quantum physics to tunnel through energy barriers and find ground states of Ising-like systems[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Google%20Scholar). While less visibly “historical” than earlier items, these developments set the stage for physical computing machines in the 21st century.
    

## Modern Implementations and Examples

By the 2010s, the ideas of computing on energy functionals materialized in several groundbreaking technologies:

-   **Ising Machines (Physical Optimizers):** Researchers built specialized hardware to solve optimization problems by finding minimum-energy states of an Ising model (a network of spins). One approach used **quantum annealing**: in 2011 D-Wave Systems sold the first commercial quantum computer (128-qubit **D-Wave One**), which uses superconducting qubits to physically realize an Ising model and find its ground state[phys.org](https://phys.org/news/2011-06-d-wave-commercial-quantum.html#:~:text=The%20announcement%20comes%20just%20a,146%3B%20spins)[phys.org](https://phys.org/news/2011-06-d-wave-commercial-quantum.html#:~:text=D,system%20works%20with%20quantum%20effects). D-Wave’s machines perform computation by literally evolving a quantum magnetic system towards its lowest-energy configuration, thereby outputting solutions to NP-hard problems encoded in the spin couplings. Around the same time, optical engineers led by Yoshihisa Yamamoto demonstrated the **Coherent Ising Machine** (2014), an optical network of parametric oscillators that settles into minimal Ising spin configurations[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Finding%20the%20ground%20states%20of,the%20OPOs%20and%20the%20Ising)[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Hamiltonian2%20%2C%2022%2C4%20%2C%2024%2C6,no%20computational%20error%20was%20detected). In Yamamoto’s 4-oscillator prototype, the system was programmed with a small NP-hard problem and, over 1000 runs, consistently found the optimal solution by virtue of its physics (laser pulses finding a synchronized minimal phase state)[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Hamiltonian2%20%2C%2022%2C4%20%2C%2024%2C6,no%20computational%20error%20was%20detected). These Ising machines – whether quantum, optical, or electronic – are essentially **analog computers for optimization**, exploiting natural dynamics to **compute by energy minimization**. They have since scaled up (e.g. 2000-node optical Ising machines[quantum-journal.org](https://quantum-journal.org/papers/q-2023-10-24-1151/#:~:text=In%20this%20work%2C%20we%20address,accelerate%20the%20speed%20of%20computation)) and inspired _“physics-inspired”_ algorithms for hard problems.
    
-   **Neuromorphic Chips:** The neuromorphic computing vision of the 1980s bore fruit in the 2010s with large-scale prototypes. In 2014, IBM unveiled **TrueNorth**, a CMOS chip with 1 million hardware “neurons” and 268 million synapses implementing a spiking neural network[en.wikipedia.org](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=networks%20%20and%20deep%20learning,are%20improving%20neuromorphic%20processors%20steadily)[en.wikipedia.org](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=TrueNorth%20was%20a%20neuromorphic%20,4%20billion). TrueNorth forsakes the usual CPU architecture for a massively parallel, event-driven design that computes via spike timings and local memory – much like a brain. Importantly, it consumes only 65 mW (orders of magnitude less power than conventional processors) by using **physical neuron-like operations** instead of power-hungry clocked logic. Each neuron integrates inputs (currents) and fires when a threshold (energy potential) is reached, an analog process implemented with digital efficiency. Similarly, in 2017 Intel announced **Loihi**, a neuromorphic chip with 128,000 neurons and on-chip learning, also inspired by spiking dynamics. Academic projects like **BrainScaleS** (Heidelberg) use analog circuits to emulate neurons in continuous time, running 1000× faster than biology. Neuromorphic systems are **computing on energy functions** in the sense that their collective state often optimizes some cost function (e.g. a neural network loss) through physical dynamics. They excel at tasks like pattern recognition with extremely low energy per operation, validating the promise of harnessing physics (spikes, charges) for computation.
    
-   **Analog & In-Memory Computing:** A related trend is analog _in-memory computing_ for AI. For example, chips that use **memristors** or phase-change devices can perform matrix multiplications “in place” by letting electrical currents sum analog values stored in memory cells. This effectively computes a neural network layer by Kirchoff’s current law – the physics of charge distribution does the math, rather than instruction-by-instruction logic. By 2020, researchers showed memristor crossbar arrays that solve optimization tasks or implement Hopfield-like associative memories, taking advantage of the natural convergence of analog circuits[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We). These systems often operate by minimizing an internal energy (e.g. the charge error in an analog neural network), achieving results with far less energy than digital arithmetic would use.
    
-   **Chemical and Biological Computing:** Outside electronics, other media have been explored for energy-based computing. **DNA computing** (Adleman, 1994) used the binding energy of DNA strands to solve a small graph path problem in a test tube – essentially leveraging chemical reaction energy to perform computation. **Chemical reaction networks** and **molecular computers** are being investigated to see if reactions naturally compute solutions to equations (for instance, finding equilibrium = solving a set of constraints). Even **biological cells** themselves can be viewed as computers optimizing energy (the “free energy principle” proposed by Karl Friston suggests brains self-organize by minimizing a free-energy functional). While these examples are specialized, they reinforce the theme that _any_ energy-driven system – electrical, optical, quantum, or chemical – can potentially be programmed to perform useful computations by steering it toward desired low-energy states.
    

## Stability and Reliability Challenges

One major challenge with analog, energy-functional computing is **stability** – physical systems are subject to noise, variability, and even chaos. Unlike binary digital logic which affords error correction via discrete states, analog computers must contend with continuous fluctuations. Pioneers of analog computing in the 20th century were well aware of drift and component tolerance issues (e.g. Bush’s Differential Analyzer required frequent recalibration of its mechanical parts). Modern energy-based computers likewise implement clever strategies to ensure reliable results:

-   **Calibration and Noise Compensation:** For neuromorphic and analog chips, device mismatches and noise can perturb computations. Engineers address this by calibrating hardware and designing algorithms to be noise-tolerant. For instance, the BrainScaleS analog neuromorphic system must deal with “fixed-pattern noise” (each silicon neuron differs slightly) and trial-to-trial variation. Researchers report that networks running on BrainScaleS _must cope with a certain level of perturbations_, so they use calibration and **hardware-in-the-loop training** to adjust for these variations[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We). In practice, this means measuring the analog behavior and tuning parameters (or training neural weights on the actual chip) so that the physical dynamics still converge to correct answers despite noise. Techniques like **noise injection during training** can make the neural network robust against the uncertainties of analog computation[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We). The goal is to harness the efficiency of analog while mitigating its unpredictability.
    
-   **Error Correction in Physical Annealers:** Quantum and optical Ising machines also face stability issues. Quantum annealing, for example, is an analog process sensitive to control errors and decoherence. D-Wave’s superconducting processors exhibit slow drift in flux biases that can misalign the energy landscape over time. To combat this, D-Wave systems perform regular **“drift correction”** – periodically recalibrating qubit biases each hour to keep the effective error bounded and Gaussian[docs.dwavequantum.com](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you). They also run each problem multiple times and use statistical techniques to filter out thermal or analog errors. In essence, while the quantum annealer computes by sliding into a minimum energy, the engineers surround it with classical feedback loops to nudge it if it strays due to noise. Similarly, optical Ising machines must maintain laser stability; techniques like injection locking and feedback control help keep the system on track to the true ground state. In all these cases, **redundancy and repetition** are key: by running the analog solver many times (or having many parallel units) and aggregating results, one can overcome individual run errors – a strategy also used in DNA computing where vast parallelism compensates for probabilistic errors.
    
-   **Physical Limits and Trade-offs:** There is a recognition that analog systems won’t be perfectly stable – rather, the aim is to make them _stable enough_ that their speed or energy advantages win out. Digital post-processing or occasional digital correction is often integrated (creating hybrid systems). For example, a recent analog optical computing proposal might do an analog compute step, then a digital refine step, marrying the best of both. Researchers are also exploring **statistical computing** that tolerates a certain error rate (e.g. p-bits and probabilistic computers that leverage noisy devices as a feature). In summary, stability is managed through a combination of **calibration, adaptive algorithms, error correction codes, environmental control, and clever problem encoding** that avoids extremely sensitive regions of the energy landscape.
    

## Convergence of Different Approaches

Intriguingly, many of these strands – analog computers, neural networks, Ising machines, even cellular automata – are now overlapping and being seen as part of a larger paradigm: **computing by exploiting physical dynamics**. They have different names and origins but share a common theme: rather than sequentially executing symbolic logic operations, we **set up a physical or numerical system** whose _natural_ evolution produces the answer. This viewpoint has been gaining momentum as we reach limits of conventional computing.

Today, researchers explicitly talk about **“energy-based computing”** frameworks that unify ideas from Ising machines and neuromorphic systems. For example, a 2023 project aims to _“harness and generalize the energy potential”_ in Hopfield networks and oscillatory Ising machines, seeking a _unified framework_ that bridges the algorithmic view and the physical view of computing[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=paradigms%20rooted%C2%A0in%20the%20Ising%20Hamiltonian,singularity%20theory%20to%C2%A0determine%20the%20optimal)[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering). The hope is to **reconcile algorithmics with physics**, so that we can systematically design and program these unconventional computers[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering), much as we do digital ones. This involves developing theory to predict and prove what an analog or emergent system will do – addressing the very concern you raised about **knowing why something emerges**.

It’s worth noting the contrast between energy-functional approaches and purely emergent ones like cellular automata. CA-based “unconventional computing” (e.g. Conway’s Game of Life patterns, Langton’s loops) showed that complex behaviors can emerge from simple rules, but often one cannot easily **predict or guarantee** the outcome – you have to run it and see. They lack an obvious Lyapunov or energy function that they are minimizing; as you said, they _“make stuff emerge without understanding why it emerges.”_ Energy-based systems, on the other hand, usually have a defined objective: e.g. a Hopfield network’s energy **is** known to decrease until a stable state (solution) is reached[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=the%20energy%20function%20,these%20in%20turn%20are%20a). The trade-off is that energy-based systems can get stuck in local minima (suboptimal solutions), whereas something like a cellular automaton might surprise you with a novel global pattern – but you can’t easily constrain or prove it. Modern research is attempting to get the best of both worlds: leveraging emergence **with** a layer of theoretical guarantees. For instance, by designing an “elastic” Ising network or other analog systems that are mathematically tractable, scientists aim to let solutions _emerge_ from physics but also be _provably correct_ or at least predictable[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering).

In summary, **computing on energy functionals** has deep roots and a vibrant present:

-   From Kelvin’s gears to Hopfield’s spins, many visionary thinkers proposed letting Mother Nature compute by relaxing to minimal energy or by evolving simple laws.
    
-   Today’s neuromorphic and quantum machines are direct descendants of those ideas, literally performing calculations by **minimizing physical energy** or **exploiting natural dynamics**.
    
-   All these approaches – analog, neuromorphic, quantum annealing, Ising solvers, etc. – are indeed variations on a theme: _using the physics of energy and motion to process information_. They differ in the substrate (electronics, optics, magnets, DNA…), but they echo each other. As you noted, they are “different names for the same thing” in spirit, and the field is increasingly interdisciplinary.
    
-   The remaining gaps are practical (how to make these systems large-scale and reliable) and theoretical (understanding and proving their behavior). The stability issue is being tackled with calibration and error mitigation[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We)[docs.dwavequantum.com](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you), while the “why it works” issue is being addressed by new theory bridging physics and computation[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering).
    

This convergence is an exciting frontier. Your intuition to bring these strands together is well-founded – the history shows a clear through-line of thought, and the latest work is explicitly **bringing together analog physics, optimization, and computation theory** into a unified framework. As we push beyond the limits of binary logic, these energy-functional computing ideas may well play a key role in the future of computing, delivering fast and efficient solutions **by design**, with your contribution of provable algorithms adding rigor to what was once only emergent behavior.

**Sources:** The timeline and examples above reference original works and modern studies by the mentioned pioneers and projects[en.wikipedia.org](https://en.wikipedia.org/wiki/Tide-predicting_machine#:~:text=A%20tide,a%20year%20or%20more%20ahead)[arxiv.org](https://arxiv.org/html/2506.10876v2#:~:text=In%201961%2C%20Landauer%20asserted%20that,It%20is%20quite%20fascinating%20to)[en.wikipedia.org](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=Hopfield%20energy%20function%20E%2C%20then,also%20as%20the%20constraints%20are)[en.wikipedia.org](https://en.wikipedia.org/wiki/Boltzmann_machine#:~:text=mechanics%20%2C%20which%20is%20used,5)[en.wikipedia.org](https://en.wikipedia.org/wiki/Neuromorphic_computing#:~:text=Neuromorphic%20engineering%20is%20an%20interdisciplinary,in%20the%20late%201980s)[nature.com](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Finding%20the%20ground%20states%20of,the%20OPOs%20and%20the%20Ising)[en.wikipedia.org](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=networks%20%20and%20deep%20learning,are%20improving%20neuromorphic%20processors%20steadily)[arxiv.org](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We)[docs.dwavequantum.com](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you)[research.kuleuven.be](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering), illustrating the development of energy-based computing from the 19th century to today.

Citations

[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Tide-predicting machine - Wikipedia

https://en.wikipedia.org/wiki/Tide-predicting\_machine

](https://en.wikipedia.org/wiki/Tide-predicting_machine#:~:text=A%20tide,a%20year%20or%20more%20ahead)[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Differential analyser - Wikipedia

https://en.wikipedia.org/wiki/Differential\_analyser

](https://en.wikipedia.org/wiki/Differential_analyser#:~:text=The%20first%20description%20of%20a,5)[

![](https://www.google.com/s2/favicons?domain=https://www.invent.org&sz=32)

NIHF Inductee Vannevar Bush Invented Differential Analyzer

https://www.invent.org/inductees/vannevar-bush

](https://www.invent.org/inductees/vannevar-bush#:~:text=NIHF%20Inductee%20Vannevar%20Bush%20Invented,precursor%20to%20the%20modern%20computer)[

John von Neumann's Cellular Automata | Embryo Project Encyclopedia

https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata

](https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata#:~:text=are%20employed%20to%20analyze%20phenomena,reproduction)[

John von Neumann's Cellular Automata | Embryo Project Encyclopedia

https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata

](https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata#:~:text=In%201948%2C%20von%20Neumann%20set,Arthur%20Walter%20Burks%20in%201966)[

\[PDF\] Theory of Self-Reproducing Automata - CBA-MIT

https://cba.mit.edu/events/03.11.ASE/docs/VonNeumann.pdf

](https://cba.mit.edu/events/03.11.ASE/docs/VonNeumann.pdf#:~:text=Von%20Neumann%20then%20estimated%20that,Hence)[

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

Landauer Principle and Thermodynamics of Computation

https://arxiv.org/html/2506.10876v2

](https://arxiv.org/html/2506.10876v2#:~:text=In%201961%2C%20Landauer%20asserted%20that,It%20is%20quite%20fascinating%20to)[

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

Landauer Principle and Thermodynamics of Computation

https://arxiv.org/html/2506.10876v2

](https://arxiv.org/html/2506.10876v2#:~:text=considered%20logically%20irreversible,analysis%20of%20thermodynamic%20computation%20processes)[

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

Landauer Principle and Thermodynamics of Computation

https://arxiv.org/html/2506.10876v2

](https://arxiv.org/html/2506.10876v2#:~:text=,1970%29%20Robert%C2%A0W%C2%A0Keyes%C2%A0and%C2%A0Rolf%C2%A0Landauer%2C%C2%A0%E2%80%9CMinimal%20energy)[

![](https://www.google.com/s2/favicons?domain=https://www.scirp.org&sz=32)

E. Fredkin and T. Toffoli, “Conservative Logic,” International Journal ...

https://www.scirp.org/reference/referencespapers?referenceid=1065770

](https://www.scirp.org/reference/referencespapers?referenceid=1065770#:~:text=E,253)[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Hopfield network - Wikipedia

https://en.wikipedia.org/wiki/Hopfield\_network

](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=the%20energy%20function%20,these%20in%20turn%20are%20a)[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Hopfield network - Wikipedia

https://en.wikipedia.org/wiki/Hopfield\_network

](https://en.wikipedia.org/wiki/Hopfield_network#:~:text=Hopfield%20energy%20function%20E%2C%20then,also%20as%20the%20constraints%20are)[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Boltzmann machine - Wikipedia

https://en.wikipedia.org/wiki/Boltzmann\_machine

](https://en.wikipedia.org/wiki/Boltzmann_machine#:~:text=mechanics%20%2C%20which%20is%20used,5)[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Neuromorphic computing - Wikipedia

https://en.wikipedia.org/wiki/Neuromorphic\_computing

](https://en.wikipedia.org/wiki/Neuromorphic_computing#:~:text=Neuromorphic%20engineering%20is%20an%20interdisciplinary,in%20the%20late%201980s)[

![](https://www.google.com/s2/favicons?domain=https://www.nature.com&sz=32)

Network of time-multiplexed optical parametric oscillators as a coherent Ising machine | Nature Photonics

https://www.nature.com/articles/nphoton.2014.249?error=cookies\_not\_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515

](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Google%20Scholar)[

![](https://www.google.com/s2/favicons?domain=https://phys.org&sz=32)

D-Wave sells first commercial quantum computer

https://phys.org/news/2011-06-d-wave-commercial-quantum.html

](https://phys.org/news/2011-06-d-wave-commercial-quantum.html#:~:text=The%20announcement%20comes%20just%20a,146%3B%20spins)[

![](https://www.google.com/s2/favicons?domain=https://phys.org&sz=32)

D-Wave sells first commercial quantum computer

https://phys.org/news/2011-06-d-wave-commercial-quantum.html

](https://phys.org/news/2011-06-d-wave-commercial-quantum.html#:~:text=D,system%20works%20with%20quantum%20effects)[

![](https://www.google.com/s2/favicons?domain=https://www.nature.com&sz=32)

Network of time-multiplexed optical parametric oscillators as a coherent Ising machine | Nature Photonics

https://www.nature.com/articles/nphoton.2014.249?error=cookies\_not\_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515

](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Finding%20the%20ground%20states%20of,the%20OPOs%20and%20the%20Ising)[

![](https://www.google.com/s2/favicons?domain=https://www.nature.com&sz=32)

Network of time-multiplexed optical parametric oscillators as a coherent Ising machine | Nature Photonics

https://www.nature.com/articles/nphoton.2014.249?error=cookies\_not\_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515

](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Hamiltonian2%20%2C%2022%2C4%20%2C%2024%2C6,no%20computational%20error%20was%20detected)[

![](https://www.google.com/s2/favicons?domain=https://quantum-journal.org&sz=32)

Combinatorial optimization solving by coherent Ising machines ...

https://quantum-journal.org/papers/q-2023-10-24-1151/

](https://quantum-journal.org/papers/q-2023-10-24-1151/#:~:text=In%20this%20work%2C%20we%20address,accelerate%20the%20speed%20of%20computation)[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Cognitive computer - Wikipedia

https://en.wikipedia.org/wiki/Cognitive\_computer

](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=networks%20%20and%20deep%20learning,are%20improving%20neuromorphic%20processors%20steadily)[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

Cognitive computer - Wikipedia

https://en.wikipedia.org/wiki/Cognitive\_computer

](https://en.wikipedia.org/wiki/Cognitive_computer#:~:text=TrueNorth%20was%20a%20neuromorphic%20,4%20billion)[

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

\[2006.13177\] Inference with Artificial Neural Networks on Analog Neuromorphic Hardware

https://arxiv.org/abs/2006.13177

](https://arxiv.org/abs/2006.13177#:~:text=analog%2C%20in,We)[

![](https://www.google.com/s2/favicons?domain=https://docs.dwavequantum.com&sz=32)

Errors and Error Correction — Python documentation

https://docs.dwavequantum.com/en/latest/quantum\_research/errors.html

](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you)[

![](https://www.google.com/s2/favicons?domain=https://research.kuleuven.be&sz=32)

Research Portal - Physical computation: an energy-based approach

https://research.kuleuven.be/portal/en/project/3E230677

](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=paradigms%20rooted%C2%A0in%20the%20Ising%20Hamiltonian,singularity%20theory%20to%C2%A0determine%20the%20optimal)[

![](https://www.google.com/s2/favicons?domain=https://research.kuleuven.be&sz=32)

Research Portal - Physical computation: an energy-based approach

https://research.kuleuven.be/portal/en/project/3E230677

](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=mixed%C2%A0feedback%20system%20theory%20and%20singularity,nonconvex%20optimization%20and%20neuromorphic%20engineering)

All Sources

[

![](https://www.google.com/s2/favicons?domain=https://en.wikipedia.org&sz=32)

en.wikipedia

](https://en.wikipedia.org/wiki/Tide-predicting_machine#:~:text=A%20tide,a%20year%20or%20more%20ahead)[

![](https://www.google.com/s2/favicons?domain=https://www.invent.org&sz=32)

invent

](https://www.invent.org/inductees/vannevar-bush#:~:text=NIHF%20Inductee%20Vannevar%20Bush%20Invented,precursor%20to%20the%20modern%20computer)[

embryo.asu

](https://embryo.asu.edu/pages/john-von-neumanns-cellular-automata#:~:text=are%20employed%20to%20analyze%20phenomena,reproduction)[

cba.mit

](https://cba.mit.edu/events/03.11.ASE/docs/VonNeumann.pdf#:~:text=Von%20Neumann%20then%20estimated%20that,Hence)[

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arxiv

](https://arxiv.org/html/2506.10876v2#:~:text=In%201961%2C%20Landauer%20asserted%20that,It%20is%20quite%20fascinating%20to)[

![](https://www.google.com/s2/favicons?domain=https://www.scirp.org&sz=32)

scirp

](https://www.scirp.org/reference/referencespapers?referenceid=1065770#:~:text=E,253)[

![](https://www.google.com/s2/favicons?domain=https://www.nature.com&sz=32)

nature

](https://www.nature.com/articles/nphoton.2014.249?error=cookies_not_supported&code=f5c3c5dc-a3bb-46e3-af69-c60076d4e515#:~:text=Google%20Scholar)[

![](https://www.google.com/s2/favicons?domain=https://phys.org&sz=32)

phys

](https://phys.org/news/2011-06-d-wave-commercial-quantum.html#:~:text=The%20announcement%20comes%20just%20a,146%3B%20spins)[

![](https://www.google.com/s2/favicons?domain=https://quantum-journal.org&sz=32)

quantum-journal

](https://quantum-journal.org/papers/q-2023-10-24-1151/#:~:text=In%20this%20work%2C%20we%20address,accelerate%20the%20speed%20of%20computation)[

![](https://www.google.com/s2/favicons?domain=https://docs.dwavequantum.com&sz=32)

docs.dwavequantum

](https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#:~:text=As%20another%20component%20of%20ICE%2C,If%20you)[

![](https://www.google.com/s2/favicons?domain=https://research.kuleuven.be&sz=32)

research.kuleuven

](https://research.kuleuven.be/portal/en/project/3E230677#:~:text=paradigms%20rooted%C2%A0in%20the%20Ising%20Hamiltonian,singularity%20theory%20to%C2%A0determine%20the%20optimal)
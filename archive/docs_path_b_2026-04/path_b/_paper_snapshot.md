A Protocol-Mediated Physics-Informed Interoperable Multi-Agent Framework
for Explainable Perishable Supply Chains

Nahid Sarker ^a^, Monzure-Khoda Kazi ^a,\*^

^a^ Karen M. Swindler Department of Chemical and Biological Engineering,
South Dakota School of Mines and Technology, Rapid City, SD, 57701, USA

Abstract

Perishable agri-food supply chains remain vulnerable to disruption
because sensing, forecasting, compliance, and routing decisions are
typically optimized in isolation rather than coordinated through a
common decision interface. This paper presents AGRI-BRAIN, a
physics-informed multi-agent framework for resilient perishable supply
chains in which communication is treated as an explicit decision
variable rather than a passive information layer. The framework
integrates a physics-informed neural network for spoilage estimation, a
socially informed routing policy, and a three-channel interoperability
architecture comprising protocol-mediated tool access, physics-informed
retrieval-augmented knowledge integration, and blockchain-backed
governance. Its key methodological contribution is a context-injection
mechanism that maps external regulatory, operational, and historical
signals into the routing policy through a learned logit-level modifier,
allowing heterogeneous information sources to directly alter action
selection. The framework is evaluated on a 72-hour spinach cold-chain
case study with five supply-chain agents under thermal, supply,
infrastructure, and market disruptions across 800 simulation episodes.
Relative to isolated decision baselines, AGRI-BRAIN improves the
Adaptive Resilience Index by up to 80%, reduces operational waste by
82%, and lowers transport emissions by 57%, while ablation studies show
that protocol-mediated tools and retrieval-based knowledge each provide
non-redundant gains. These results show that the main bottleneck in
autonomous perishable logistics is not isolated model accuracy alone,
but the absence of standardized and verifiable mechanisms for
inter-agent coordination. AGRI-BRAIN provides a process-systems-oriented
architecture for explainable, auditable, and sustainability-aware
supply-chain decision-making.

*Keywords:* Agentic AI; Multi-agent Intelligent Systems; Supply-chain
Interoperability; Retrieval-Augmented Generation; Explainable Decision
Auditing; Reverse Logistics.

\* Corresponding author.

*E-mail*: Kazi.Khoda\@sdsmt.edu (Monzure-Khoda Kazi), *Phone:* +1 979
721 0640.

# Introduction

Perishable agri-food supply chains are difficult to manage because the
value of a decision depends not only on cost or routing distance, but
also on quality degradation, timing, compliance, recovery capacity, and
downstream social consequences. In cold-chain operations, product
quality can deteriorate irreversibly within hours, so decisions that are
computationally acceptable in conventional supply chains may already be
too late for perishables. A routing change that would be reasonable for
durable goods may be ineffective for fresh produce if it does not
account for spoilage kinetics, refrigeration performance, regulatory
limits, and the absorption capacity of redistribution or recovery
pathways. The central challenge is therefore not only prediction or
optimization in isolation, but real-time coordination across multiple
decision layers under shrinking intervention windows.

Process systems engineering has long emphasized the importance of
coordinated decision-making across supply chains and enterprise-wide
systems, particularly when local decisions create network-wide
consequences (Grossmann, 2005, 2012). This perspective is especially
relevant to perishable logistics, where upstream sensing and forecasting
must influence downstream allocation, transport, and recovery decisions
before quality losses become irreversible. Yet agri-food supply chains
remain highly fragmented in practice. Quality monitoring, compliance
checks, demand forecasting, routing optimization, and traceability
systems are often implemented as separate modules, with communication
occurring through dashboards, manual review, or loosely coupled
summaries rather than through machine-actionable interfaces. This
fragmentation contributes to delayed interventions, brittle responses
under disruption, and limited accountability for autonomous decisions.
The practical importance of the problem is clear: approximately 14% of
food is lost between harvest and retail globally (FAO, 2019), and
disruptions in food supply chains can propagate rapidly across
producers, processors, distributors, and recovery organizations, as
highlighted during the COVID-19 period (Swinnen and McDermott, 2020).
More broadly, this problem sits at the emerging intersection of process
systems engineering and agentic AI.

Recent work has improved several components of this broader problem.
Centralized and distributed optimization methods have advanced
supply-chain coordination and enterprise-wide decision-making
(Grossmann, 2012). Machine learning methods have improved forecasting,
classification, and operational support for process and supply systems
(Pistikopoulos et al., 2021). Multi-agent systems have provided useful
abstractions for distributed decision-making in complex industrial
environments (Leitão, 2009). Blockchain-based approaches have improved
provenance, transparency, and transaction integrity in supply chains
(Saberi et al., 2019; Kazi and Hasan, 2024). However, these advances do
not fully resolve a key systems problem in perishable logistics: how
heterogeneous agents, models, tools, and institutional knowledge sources
should communicate so that information generated at one point in the
chain can directly and verifiably alter downstream decisions. Recent
advances in autonomous AI agents have made it increasingly feasible to
connect sensing, tool use, retrieval, and action in distributed
operational environments. Yet, in perishable supply chains, such agents
are only useful if their decisions remain physically plausible,
institutionally informed, and verifiable across organizational
boundaries. Taken together, these observations point to a missing
decision layer: a structured mechanism through which heterogeneous
agents and external knowledge sources can exchange context in a form
that directly enters action selection rather than remaining external to
it.

The first unresolved gap is therefore **interoperability at the decision
level**. Existing approaches often connect forecasting, compliance
checking, optimization, and traceability components through ad hoc
integration, centralized orchestration, or post-processed information
exchange. In many cases, a spoilage forecast, a compliance warning, a
regulatory document, or a recovery-capacity update may exist somewhere
in the digital stack, but not in a representation that can
systematically enter another agent's policy computation. As a result,
supply-chain intelligence remains modular rather than truly coordinated.
For perishable systems, this gap is costly because the decision
context is inherently distributed across physical measurements,
operational rules, historical precedents, and external constraints. A
routing policy that only reads local state variables cannot naturally
represent whether a temperature excursion constitutes a regulatory
breach, whether a redistribution partner has remaining capacity, or
whether recent decisions have already saturated a downstream recovery
pathway. In this sense, interoperability is not merely a
software-integration problem; it is the mechanism that determines
whether distributed information can become operationally actionable.

The second unresolved gap is **structural explainability**.
Explainability in AI-enabled logistics is commonly treated as a post hoc
activity, in which a separate interpretation layer is applied after a
decision has been produced (Barredo Arrieta et al., 2020). Such
explanations may be descriptively useful, but they are often only
loosely connected to the actual internal computation that generated the
action. In settings such as food logistics, where decisions can affect
waste generation, food access, and regulatory exposure, explanation must
be more than a narrative add-on. It must be traceable to the same
contextual signals and transformations that shaped the decision itself.
This requirement is closely linked to interoperability: once external
tool outputs, retrieved documents, and agent histories are converted
into a structured decision representation, they can serve not only as
policy inputs but also as the basis of a causal evidence trail.
Structural explainability therefore does not arise independently of
coordination; it becomes achievable because coordinated,
machine-actionable communication makes the decision pathway inspectable
by construction. This need also aligns with broader concerns in process
systems engineering regarding purely data-driven AI, especially in
scientific and engineering domains where feasible decisions must
remain consistent with physical knowledge, constraints, and
domain-specific logic (Venkatasubramanian, 2019; Venkatasubramanian
and Chakraborty, 2025).

A third related issue concerns how **nonlocal decision objectives**,
including sustainability-relevant criteria, enter the decision
architecture. Much of the optimization literature on sustainable supply
chains still emphasizes economic and environmental metrics more strongly
than social criteria, and social life cycle assessment indicators are
infrequently used as active drivers of optimization rather than ex post
reporting metrics (Barbosa-Póvoa et al., 2018). This is particularly
important in perishable reverse logistics, where redistribution and
recovery decisions affect not only waste and emissions but also food
access, labor implications, and community resilience (Kazancoglu et al.,
2021; Teigiserova et al., 2020). The point here is not simply that
sustainability should be "included" as an additional module. Rather,
sustainability metrics represent a broader class of contextual
information that is difficult to operationalize in isolated routing
policies because it is distributed, cross-cutting, and often external to
immediate physical state measurements. A resilient perishable
supply-chain architecture should therefore be able to incorporate such
information alongside compliance signals, operational constraints, and
historical precedents, so that these factors can meaningfully influence
action selection rather than remain passive reporting outputs. At the
same time, the growing use of AI raises an additional design
consideration: the computational footprint of digital decision systems
should itself be made transparent, consistent with the broader Green AI
agenda (Schwartz et al., 2020). Taken together, these issues suggest
that resilient perishable supply chains require more than better
prediction accuracy; they require a decision architecture that can
combine physical state estimation, cross-agent communication,
institutional knowledge, and broader operational objectives within a
unified computational framework.

To address this need, this paper presents **AGRI-BRAIN**, a
protocol-mediated multi-agent framework for perishable agri-food supply
chains. The framework is built on the premise that communication should
be treated as an explicit component of the decision architecture rather
than as a passive integration layer. AGRI-BRAIN represents five
supply-chain stages through heterogeneous agents and links them through
three complementary communication channels: direct structured agent
messages, protocol-mediated tool access, and physics-informed retrieval
of institutional knowledge. Information from these channels is
transformed into a structured context representation that directly
modifies routing decisions through a learned policy-level mapping. In
this way, external context such as compliance status, spoilage urgency,
recovery saturation, and retrieved operational guidance becomes
actionable within the policy itself rather than remaining external to
it.

The framework couples this communication architecture with a
physics-informed spoilage estimator, a routing objective that accounts
for operational and sustainability-relevant criteria, and
blockchain-supported decision logging and governance. The spoilage
module provides physically plausible quality estimates under variable
thermal conditions; the routing layer accounts for redistribution and
broader supply-chain objectives alongside operational performance; and
the governance layer provides auditable provenance for policy-relevant
evidence and decisions. The intended contribution of the paper is
therefore methodological rather than merely integrative. The main claim
is not simply that multiple digital components can be assembled into a
single platform, but that standardized and verifiable communication
among heterogeneous decision components can produce measurable gains in
resilience, waste reduction, explainability, and accountability relative
to isolated decision architectures.

The specific contributions of this work are fourfold. First, we develop
a **three-channel, protocol-mediated interoperability framework**
through which peer-agent messages, external tool outputs, and retrieved
institutional knowledge are converted into a unified context
representation for downstream action selection. Second, we propose a
**physics-informed retrieval and context-injection mechanism** that
links operational state to the selection, weighting, and policy-level
use of external knowledge in the decision process. Third, we design a
**structural explainability mechanism** in which the same contextual
features that modify the policy also generate the causal evidence trail
used for explanation and audit. Fourth, we incorporate **social and
governance-relevant decision signals** into the routing architecture so
that sustainability-related criteria, provenance, and coordination
constraints can influence action selection as active decision variables
rather than passive reporting outputs.

The framework is evaluated on a fresh spinach cold-chain case study
representing a South Dakota agricultural cooperative. The study
considers five supply-chain nodes and multiple disruption scenarios
involving thermal stress, supply surplus, infrastructure failure, and
market volatility. Performance is assessed through controlled ablations,
repeated stochastic simulations, and metrics including waste, emissions,
social performance, resilience, and robustness under communication
faults. Across these tests, the paper evaluates whether the principal
bottleneck in intelligent perishable logistics lies not only in isolated
model quality, but in the absence of a standardized mechanism for
transforming heterogeneous information into coordinated, explainable,
and auditable decisions.

Beyond the methodological contributions, the framework is implemented as
a complete operational platform with a React and FastAPI frontend
providing real-time dashboards for cold-chain telemetry, decision traces
with causal explanations, model context protocol (MCP) and
physics-informed retrieval-augmented generation (piRAG) monitoring
panels, and an optional blockchain verification of every routing
decision. The architecture supports three deployment phases: monitoring,
advisory with operator override, and fully autonomous execution. The
claims are validated across five disruption scenarios through a
controlled ablation design with 20 independently seeded replications,
four communication fault categories, and explainability assessment
through causal chain coverage, sign consistency, and provenance
integrity. The complete codebase and all evaluation artifacts are
publicly available\*\*.

The remainder of the paper is organized as follows. Section 2 defines
the system requirements and design principles. Section 3 presents the
AGRI-BRAIN methodology, including the perception models, communication
architecture, routing policy, and governance layer. Section 4 reports
the case-study results, ablation analyses, and robustness tests. Section
5 concludes with implications and discussion.

# System Requirements and Design Principles of AGRI-BRAIN

The gaps identified in Section 1 imply that an intelligent perishable
supply-chain architecture must satisfy several coupled design
requirements rather than optimize a single objective in isolation. In
perishable agri-food systems, interoperability, resilience,
reverse-logistics responsiveness, explainability, governance, and
sustainability are interdependent and may conflict in practice. For
example, preserving freshness may favor rapid cold-chain transport,
whereas redistribution, lower emissions, or broader community benefit
may favor different routing actions. Similarly, increasing autonomy
without traceable decision logic can improve speed while weakening
accountability. The purpose of this section is therefore to translate
the conceptual gaps established in Section 1 into concrete design
requirements that guide the AGRI-BRAIN architecture presented in Section
3.

These requirements are not introduced as abstract aspirations. They
arise directly from the operational realities of perishable logistics
and from the architectural claim of this paper: that the principal
bottleneck is not isolated model quality alone, but the absence of a
structured mechanism through which heterogeneous agents, tools, and
institutional knowledge can contribute to coordinated and explainable
decisions. Accordingly, the AGRI-BRAIN framework is designed around six
system principles: adaptive decision-making under disruption,
distributed coordination and interoperability, circular reverse
logistics as a first-class decision mode, auditable governance and
decision provenance, embedded sustainability and social accountability,
and transparent computational footprint. Together, these principles
define what the architecture must achieve before the detailed
methodology is specified.

## Adaptive Decision-Making under Disruption

Perishable supply chains operate under tight temporal constraints and
are highly sensitive to thermal excursions, demand shifts,
infrastructure failures, and market volatility. Under such conditions,
static planning assumptions and deterministic operating rules become
brittle. A viable architecture must therefore support decision-making
that remains responsive under changing regimes while preserving physical
plausibility and operational interpretability.

Three requirements follow. First, spoilage risk must be estimated
through a physics-grounded model that accounts for temperature-dependent
degradation dynamics, so that quality predictions remain plausible even
under conditions not well represented in calibration data. Second,
short-horizon demand and supply conditions must be forecast using
computationally lightweight models that can be updated frequently and
deployed close to the operational layer. Third, routing decisions must
respond to regime changes quickly enough to remain useful within narrow
perishability windows, while still allowing the basis of those decisions
to be inspected and audited. In AGRI-BRAIN, this principle motivates the
perception layer and regime-aware routing policy described in Section 3.

## Distributed Coordination and Interoperability

A perishable supply chain is not governed by a single homogeneous
objective. Farmers, processors, distributors, cooperatives, and recovery
organizations operate with different incentives, constraints, and
decision horizons. Under disruption, the effectiveness of one node's
action depends strongly on what downstream nodes can absorb, what
regulations permit, and what the broader system has already done. An
intelligent architecture must therefore support distributed
decision-making without allowing the system to fragment into isolated
local optimizers.

The key requirement is interoperability at the decision level.
Forecasts, compliance checks, routing recommendations, historical
decisions, and institutional guidance must be exchangeable in a form
that can directly alter downstream action selection. This is more
demanding than software connectivity alone. It requires that
heterogeneous outputs be converted into structured, machine-actionable
context rather than remaining external annotations or dashboard
information. In practical terms, downstream agents must be able to act
not only on local measurements, but also on signals such as spoilage
alerts, capacity updates, compliance status, and retrieved operational
guidance. This requirement motivates the multi-agent communication
architecture of AGRI-BRAIN, in which peer-agent messages, tool outputs,
and retrieved institutional knowledge are fused into a unified decision
context.

## Circular Reverse Logistics as a First-Class Decision Mode

Conventional logistics models often treat reverse flows as secondary
actions triggered only after forward fulfillment has failed. That logic
is inadequate for perishable systems. By the time waste is formally
recognized, quality loss may already be irreversible and redistribution
opportunities may have been missed. A perishable supply-chain
architecture must therefore represent redistribution and recovery as
first-class routing actions within the same decision process that
governs conventional cold-chain continuation.

This requirement has two implications. First, routing decisions must
evaluate forward distribution, local redistribution, and recovery
options within a common decision framework, rather than isolating waste
management in a separate downstream module. Second, upstream actions
must be informed by downstream absorption capacity before quality
thresholds are breached. In other words, reverse logistics must become
anticipatory rather than reactive. This design principle is central to
AGRI-BRAIN because much of the claimed value of cross-agent coordination
lies in shifting at-risk product toward equitable redistribution or
recovery before spoilage translates into avoidable waste.

## Auditable Governance and Decision Provenance

As autonomy increases, speed and adaptability improve, but so does the
need for enforceable coordination rules and traceable decision records.
In food logistics, routing decisions can affect regulatory exposure,
community access, waste generation, and incentive distribution across
organizational boundaries. An intelligent architecture must therefore do
more than recommend actions; it must also provide institutional
mechanisms through which agent participation, policy updates, and
decision evidence can be authenticated, versioned, and audited.

The core principle here is auditable governance rather than traceability
alone. Provenance records are valuable, but they are insufficient if
they do not also constrain behavior. The architecture must support
authenticated participation, tamper-evident policy parameters,
verifiable decision histories, and explicit linkage between
policy-relevant evidence and final action. These capabilities enable
explanation to function as a verifiable consequence of the architecture
rather than as a post hoc narrative. In AGRI-BRAIN, this principle
motivates the governance substrate in which routing evidence, policy
parameters, and coordination rules are recorded through
blockchain-supported mechanisms. Blockchain is therefore not treated as
the conceptual centerpiece of the framework, but as one implementation
mechanism for auditable coordination and provenance.

## Embedded Sustainability and Social Accountability

For perishable agri-food systems, decision quality cannot be judged by
logistics efficiency alone. Routing choices influence not only cost,
lead time, and spoilage, but also food access, labor implications,
community resilience, and environmental burden. If such factors remain
outside the decision architecture, optimization will tend to privilege
narrowly local performance even when broader system outcomes
deteriorate. A resilient architecture must therefore allow
sustainability-relevant and socially meaningful criteria to function as
active decision signals rather than passive reporting outputs.

This principle should not be interpreted as a claim that sustainability
integration alone is the main novelty of AGRI-BRAIN. Rather, it serves a
deeper architectural purpose. Social life cycle assessment variables,
community-oriented objectives, and redistribution priorities represent
nonlocal forms of decision context that isolated routing policies
struggle to operationalize because they are cross-cutting,
multi-criteria, and often external to immediate physical state
measurements. Embedding such signals into the routing objective is
therefore a test of whether the architecture can make heterogeneous,
distributed information actionable. In AGRI-BRAIN, this requirement
motivates the inclusion of sustainability-weighted and
redistribution-aware criteria within the policy layer so that social
accountability is reflected in action selection itself.

## Transparent Computational Footprint

AI-enabled decision support can improve resilience and efficiency, but
it can also shift environmental burden upstream to computation, storage,
and supporting digital infrastructure. In a framework that explicitly
values sustainability and accountability, the computational cost of
intelligence should not remain hidden. An additional design requirement
is therefore transparency in the digital footprint of the architecture
itself.

This principle requires that inference-time energy and water consumption
be estimated, that lightweight computational paths be preferred when
possible, and that governance mechanisms avoid unnecessarily costly
digital operations. The objective is not to eliminate computational
overhead, but to ensure that improvements in operational logistics are
not presented without acknowledging the resource cost of the decision
system that produces them. In AGRI-BRAIN, this motivates explicit
reporting of the computational footprint alongside the operational
metrics used to evaluate waste, emissions, and resilience.

Taken together, these six principles define the design requirements that
shape the AGRI-BRAIN methodology. They imply that the architecture must
combine physically grounded perception, distributed coordination,
retrieval- and tool-mediated contextual reasoning, traceable governance,
and multi-criteria decision logic within a unified system. Section 3
translates these principles into the technical framework, comprising a
perception layer (Section 3.3-3.4), a multi-agent coordination protocol
(Section 3.5-3.7), a regime-aware routing policy (Section 3.9), and a
blockchain governance substrate (Section 3.15).

# High-Level Architecture and Technical Framework

The methodology proceeds in four layers that track how information
becomes action. A **perception layer** estimates the physical state of
the supply chain from telemetry. A **communication layer** exposes
heterogeneous institutional signals to every agent through three
parallel channels: direct messages, protocol-mediated tools, and
physics-informed retrieval. An **action layer** fuses perception and
context into logit-level routing decisions through a regime-aware
contextual softmax policy. A **governance layer** constrains, evaluates,
and audits the entire pipeline through stochastic perturbation,
shared-seed ablation, paired statistics, and on-chain smart contracts.
The subsection numbering below follows this flow. Figure 1 illustrates
the complete AGRI-BRAIN decision pipeline, mapping the flow of sensor
telemetry through the perception models, the three-channel communication
layer, the contextual policy, and the blockchain governance substrate.

## Methodological Premise and Hypotheses

The central claim of this work is that the bottleneck in intelligent
agri-food supply chain management is not isolated model quality, but how
heterogeneous models communicate and coordinate under disruption.
Tightening the spoilage estimator or the demand forecaster in isolation
improves those components in isolation; it does not equip the routing
policy to recognize a regulatory violation, synthesize information
across lifecycle stages, or justify a decision to an auditor. AGRI-BRAIN
therefore treats communication as a first-class design object,
implemented through three parallel channels that fuse into a single
context representation consumed by the policy and exposed in every
explanation.

Five novel components support this premise: (i) a learned context weight
matrix $\Theta_{\text{context}}$ ∈ $\mathbb{R}^{3 \times 5\ }$ with
domain-justified sign constraints; (ii) a sign-constrained REINFORCE
update that prevents domain-violating weight flips; (iii) three
physics-informed extensions to retrieval-augmented generation
(scenario-discriminative query expansion, plausibility reranking, and a
continuity gate); (iv) a five-dimensional context feature vector ψ that
unifies MCP tool outputs and piRAG retrieval scores for both policy
consumption and explanation; and (v) a deterministic governance override
tied to the logit structure rather than to raw input thresholds. Three
falsifiable hypotheses structure the evaluation. **H1 (integration
superiority):** the full integrated architecture produces higher mean
ARI than the same models with communication disabled, with
$\Delta ARI\  \geq \ 0.005$ and adjusted $p\  < \ 0.10$. **H2
(communication robustness):** the integrated system preserves its ARI
advantage under four fault categories (sensor noise, missing data,
telemetry delay, tool fault) with \|ΔARI\| \< 0.01 per stressor. **H3
(component complementarity):** tool-only and retrieval-only ablations
each exceed the no-context control, and the integrated configuration
exceeds the better single-channel condition in every scenario.

![](media/image2.jpeg){width="6.26582239720035in"
height="6.364542869641295in"}

Figure 1. AGRI-BRAIN decision pipeline. Sensor telemetry flows through a
perception layer (PINN spoilage, LSTM demand, EWM supply), five
role-specific agents, three communication channels (direct messages, MCP
tools, piRAG retrieval), a context-modified softmax routing policy, and
evaluation models (waste, carbon, SLCA) before being logged to six
blockchain smart contracts. Dashed arrows denote feedback loops.

## Case Study and Input Data

Validating the three-channel architecture requires a testbed where
quality degradation is fast enough to demand real-time communication,
where multiple lifecycle stages create genuine coordination needs, and
where regulatory and social constraints are non-trivial. Fresh spinach
at a South Dakota agricultural cooperative satisfies all three
conditions: shelf life at ambient temperature is under 48 hours, the
supply chain spans farm, processor, cooperative, distributor, and
recovery agents over 72 hours, and FDA leafy-greens regulations,
cooperative bylaws, and food-bank protocols impose measurable
institutional constraints on routing. The simulation horizon is 72 h at
15-minute resolution (288 timesteps), with three route options (120 km
cold chain, 45 km local redistribution, 80 km recovery) calibrated to
the testbed geography. Input data comprise temperature and humidity
telemetry, an inventory time series, and a demand series perturbed by
the stochastic layer described in Section 3.12. Figure 2 presents the
full-stack architecture built to operationalize this case study,
demonstrating how the frontend dashboards, backend scientific computing
modules, and blockchain governance are organized as a fully deployable
environment.

![](media/image3.jpeg){width="6.270138888888889in"
height="3.6055555555555556in"}

Figure 2. Full-stack architecture and development ecosystem of
AGRI-BRAIN. The system integrates React-based frontend telemetry
dashboards, Python-driven backend scientific computing modules, dynamic
piRAG and MCP communication pipelines, and EVM-compatible blockchain
governance smart contracts into a unified, deployable decision
environment.

## Physics-Informed Spoilage Model

Routing decisions depend on knowing how much product quality remains and
how fast it is declining. A purely data-driven estimator requires large
training sets for every temperature profile, and a purely mechanistic
model cannot absorb residual effects beyond its closed-form form; the
spoilage model therefore couples a closed-form Arrhenius-Baranyi kinetic
baseline with a bounded neural residual trained to a composite loss.

Let $C(t)\  \in \ \lbrack 0,\ 1\rbrack$ denote the remaining quality
fraction. Fresh spinach degrades through first-order kinetics,

$\ \frac{\text{dC}}{\text{dt}}\  = \  - k_{\text{eff}}(t,\ T,\ H)\  \cdot \ C$
(1)

with Arrhenius temperature dependence and humidity coupling,

$k\ (T,\ H)\  = \text{\ k}_{\text{ref}}\  \cdot \ exp\ \lbrack E_{a}/R\  \cdot \ (1/T_{\text{ref}}\  - \ 1/T_{K})\rbrack\  \cdot \ (1\  + \ \beta\  \cdot \ a_{w})$
(2)

Parameters for fresh spinach are $\text{\ k}_{\text{ref}}$ = 0.0021 h⁻¹,
$E_{a}/R$ ≈ 66.5 kJ/mol (Giannakourou & Taoukis, 2003), $T_{\text{ref}}$
= 277.15 K, *β* = 0.25 (Labuza, 1982), and $a_{w}$ ≈ RH/100. An
early-growth suppression term accounts for the Baranyi lag phase through
a smooth sigmoid,

$\alpha(t)\  = \ t\ /\ (t\  + \ \lambda),\ \lambda\  = \ 12.0\ h$ (3)

with the lag constant *λ* = 12.0 h, typical for spinach at 4 °C (Baranyi
& Roberts, 1994), chosen so that *α*(0) = 0, *α*(*λ*) = 0.5, and *α* → 1
as *t* → ∞, so that the effective rate becomes
$\text{\ k}_{\text{eff}}\ (t,\ T,\ H)\  = \ k(T,\ H)\  \cdot \ \alpha(t)$.
The ODE is integrated along the sensor trajectory using the midpoint
rule,
$C(t_{i})\  = \ C(t_{i - 1})\  \cdot \ exp( - k_{\text{eff}}\  \cdot \ \Delta t)$,
with monotone clamping to \[0, 1\].

A physics-informed neural residual corrects closed-form errors without
discarding mechanistic grounding. A two-hidden-layer feed-forward
network $f_{\theta}$ takes the current state (*T*, RH, *ρ*, inventory
pressure) and returns a bounded correction,\
$C_{\text{PINN}}(t)\  = \ C_{\text{ODE}}(t)\  + \ 0.08\  \cdot \ tanh(f_{\theta}(x))$
(4)

trained on a composite loss that combines a data term and the ODE
residual, *\
*$\mathcal{L\  = \ }\lambda_{\text{data}}N^{- 1}\sum\Delta C^{2}\  + \ \lambda_{\text{phys}}N^{- 1}\sum r^{2}$
(5)

where $r\  = \ dC/dt\  + \ k\_\{ eff\}\  \cdot \ C$ is the residual and
$\lambda_{\text{phys}}$ = 1.0 (equal data/physics weighting). The ±0.08
tanh clamp strictly bounds the neural correction to 8% of the
mechanistic baseline, preserving physical plausibility while allowing
the network to absorb sensor drift and unmodeled effects. Spoilage risk
is $\rho(t)\  = \ 1\  - \ C(t)$, with the RLE threshold set at *ρ* =
0.10 (spoilage-meaningful deviation).

## Demand and Supply Forecasting with Regime Detection

The spoilage model estimates current quality; the routing policy also
needs to anticipate near-term conditions. Demand and supply follow
different dynamics (demand is non-stationary and regime-dependent;
supply changes smoothly with harvest and inventory cycles), so separate
forecasters are required.

The decision pipeline consumes two independent forecasts at each
timestep: a demand forecast produced by a recurrent neural network, and
a supply/yield forecast produced by exponential smoothing on the
inventory series. Separating these forecasts ensures that supply-side
and demand-side signals enter the policy as distinct features, improving
interpretability.

A standard LSTM cell with forget, input, and output gates is used: 16
hidden units, 48-step lookback (12 hours at 15-minute resolution),
per-timestep retraining (80 epochs, $lr\  = \ 0.005$). The LSTM cell
computes:

$f_{t}\  = \ \sigma(W_{f}\lbrack h_{t - 1},\ x_{t}\rbrack\  + \ b_{f}),\ \ c_{t}\  = \ f_{t}\  \odot \ c_{t - 1}\  + \ i_{t}\  \odot \ g_{t}$
(6)

with analogous expressions for $i_{t}$ (input gate), $o_{t}$ (output
gate), and $g_{t}$ (candidate cell state). The hidden state
$h_{t}\  = \ o_{t}\  \odot \ tanh(c_{t})$ is projected to a scalar
demand prediction through a linear output layer.

Supply is projected by a Holt-Winters exponential smoother
($\alpha\  = \ 0.5$, $\beta\  = \ 0.2$) applied to the inventory series.
The level and trend components update at each timestep:

> $l_{t} = \alpha\  \cdot \ y_{t}\  + \ (1\  - \ \alpha)(l_{t - 1}\  + \ b_{t - 1})$
> (7)

$b_{t} = \beta\  \cdot \ (l_{t}\  - \ l_{t - 1})\  + \ (1\  - \ \beta)\  \cdot \ b_{t - 1}$
(8)

with forecast
${\widehat{y}}_{t + h}\  = \ l_{t}\  + \ h\  \cdot \ b_{t}$. Both
forecasters operate on the perturbed telemetry stream, so forecast
errors propagate naturally through the stochastic evaluation framework.

A Bollinger-band regime detector computes a rolling z-score on the
demand series:

$z_{t} = (y_{t}\  - \ ȳ)/\sigma$ (9)

with rolling window $w = 20$ steps and threshold $|z| > 2.0$. When the
z-score exceeds the threshold, the binary regime indicator
$\tau\  = \ 1$ activates the volatility tilt in the routing policy,
shifting behavior toward conservative redistribution under shocks.

The perception layer is now complete. Each agent receives six physical
features derived from $C_{\text{PINN}}$, ŷ, inventory, temperature, and
the regime indicator, which together form the state vector
$\varphi(s)\  \in \mathbb{\ R⁶}$ consumed by the routing policy (Section
3.9). The next three subsections define the communication layer that
supplements φ(s) with institutional context.

## Multi-Agent Coordination

A single centralized controller cannot represent the competing mandates
of actors at different lifecycle stages: a farm agent prioritizes
freshness preservation, while a recovery agent prioritizes waste
valorization. Five role-specific agents are therefore instantiated, each
active over a distinct portion of the 72-h episode and each applying a
fixed logit bias that encodes its mandate (Table 1). Bias magnitudes are
constrained to \|*b*\| ≤ 0.10 so that they tilt but cannot override the
learned policy.

Table 1. Agent role assignments and logit biases.

  **Agent**     **Hours**          **Bias \[CC, LR, Rec\]**   **Mandate**
  ------------- ------------------ -------------------------- --------------------------
  Farm          0--18              \[+0.08, −0.03, −0.05\]    Preserve freshness
  Processor     18--36             \[−0.02, +0.06, −0.04\]    Processing efficiency
  Cooperative   12--30 (overlay)   \[0.00, +0.04, −0.04\]     Governance coordination
  Distributor   36--54             \[−0.05, +0.10, −0.05\]    Community redistribution
  Recovery      54+                \[−0.06, −0.02, +0.08\]    Waste valorization

Channel 1 of the communication layer provides direct inter-agent
messages through five typed classes: *spoilage alerts* (when *ρ* \>
0.30), *surplus notifications* (when inventory exceeds cold-room
capacity), *capacity updates*, *reroute requests*, and
*acknowledgments*. Structured typing lets downstream agents parse
messages deterministically without natural language processing. Channel
1 propagates information produced at one lifecycle stage to the next
agent before it becomes the active decision-maker, which is necessary
but not sufficient: agents also need structured access to external
domain-specific tools and institutional knowledge.

## Protocol-Mediated Tool Access

Agents need compliance databases, spoilage forecasters, SLCA reference
standards, recovery-capacity queries, and blockchain history that no
single agent contains internally. Channel 2 implements the Model Context
Protocol (MCP) specification over JSON-RPC 2.0, supporting three
transport options: in-process calls for simulation, standard I/O for
subprocess isolation, and HTTP for production deployment. Seven tools
are exposed: *compliance*, *spoilage_forecast*, *slca_lookup*,
*chain_query*, *recovery_capacity*, *footprint*, and *calculator*. Each
tool invocation produces a structured request (method, typed parameters,
request identifier) and receives a structured response (result object or
error with code and message). Role-specific dispatch patterns (Table 2)
reflect different information needs at different lifecycle stages.

Table 2. Role-specific MCP tool invocations per decision step.

  **Agent**     **MCP Tools Invoked**                                                       **piRAG Prompt**        **Trigger**
  ------------- --------------------------------------------------------------------------- ----------------------- ---------------
  Farm          compliance, slca_lookup, spoilage_forecast                                  Regulatory compliance   Every step
  Processor     compliance, chain_query, calculator                                         SLCA routing            Every step
  Cooperative   slca_lookup, chain_query, spoilage_forecast, footprint                      Governance policy       Overlay hours
  Distributor   compliance, slca_lookup, spoilage_forecast, recovery_capacity, calculator   Emergency rerouting     Every step
  Recovery      chain_query, slca_lookup, footprint                                         Waste hierarchy         Every step

A circuit-breaker mechanism (failure threshold = 3, reset after 5 s)
with retry and exponential backoff provides graceful degradation: the
agent continues with partial context rather than blocking the decision
pipeline. A protocol recorder captures every JSON-RPC 2.0 request and
response pair, providing verifiable evidence that Channel 2 executes
real structured communication rather than simulated stubs. The
chain_query tool in particular functions as
agent-to-agent-via-blockchain communication: the distributor effectively
learns what the farm and processor decided earlier in the episode by
reading the decision logger smart contract. Channel 2 supplies
structured numerical signals; Channel 3 supplies the institutional
knowledge required to interpret them.

## Physics-Informed Knowledge Retrieval

Tool outputs quantify what is happening; operational decision-making
also requires what should be done about it, in the form of regulatory
guidance, standard operating procedures, contingency plans, and lessons
from past interventions. Standard retrieval-augmented generation treats
all queries uniformly; a contingency plan for heatwave conditions is
irrelevant at 4 °C but critical at 14 °C. Channel 3, the
physics-informed retrieval-augmented generation (piRAG) pipeline,
addresses this with three physics-informed extensions to a hybrid BM25 +
TF-IDF retriever operating over a 20-document knowledge base (regulatory
standards, SOPs, SLCA reference documents, and contingency plans).

First, *scenario-discriminative query expansion* appends domain-specific
terms based on current physical conditions: thermal-degradation terms
when *T* \> 10 °C (the FDA leafy-greens temperature-excursion
threshold), advanced-diversion terms when *ρ* \> 0.50 (the midpoint of
the normalized spoilage axis), and exponential-acceleration terms when
$k_{\text{eff}}$ \> 0.005 h⁻¹ (approximately 2× the baseline kinetic
rate at 4 °C). Second, a *physics-plausibility reranker* adjusts passage
scores based on temperature proximity, spoilage-stage keyword matching,
and urgency alignment; the physics bonus is clamped to \[0, 0.30\] to
prevent reranker dominance over keyword match. Third, a *temporal
context window* of 20 entries (six-hour horizon) computes a continuity
score *κ* that modulates context strength through $\tau_{\text{mod}}$ =
1.3 − 0.6 *κ*, producing amplification up to 1.3× during volatile
retrieval (*κ* ≈ 0) and attenuation to 0.7× during stable periods (*κ* ≈
1). Three guard checks (dimensional analysis, feasibility, retrieval
quality) return a zero context modifier when triggered, ensuring that
bad retrieval cannot degrade decision quality below the no-context
baseline. The hybrid retriever combines BM25 keyword matching with
weight 0.6 and TF-IDF cosine similarity with weight 0.4 (selected by
grid search on a development query set, optimizing MRR\@4), returning
the top *k* = 4 passages from the corpus. A dynamic knowledge feedback
mechanism synthesizes past routing decisions into the knowledge base
every 24 timesteps (6 h), creating a blockchain-to-retrieval feedback
loop that lets institutional memory grow from agent experience. During
the first six hours of every episode, before the initial synthesis cycle
completes, retrieval relies exclusively on the static 20-document
corpus; from hour 6 onward the synthesized entries are surfaced
alongside the static documents.

Channels 2 and 3 now produce heterogeneous outputs: compliance
severities, urgency classifications, retrieval confidences, regulatory
flags, and recovery saturation estimates. The next subsection fuses them
into a single representation.

## Context Feature Extraction, Logit Integration, and Explanation

Channels 2 and 3 must be fused into a single representation that the
routing policy can consume and that is directly readable in every
explanation. Figure 3 traces these information-flow paths from the
initial agent decisions, through the external tools and retrieval
pipelines, and into the context weight matrix.

![](media/image4.png){width="6.29375in" height="2.3333333333333335in"}

**Figure 3.** Three-channel information flow and context fusion. Agent
decisions propagate through direct messages (Channel 1), MCP tool
invocations (Channel 2, producing ψ₀, ψ₁, ψ₄), and piRAG retrieval
(Channel 3, producing ψ₂, ψ₃). All five features converge into the
five-dimensional context vector (ψ) and are processed by the context
weight matrix ($\Theta_{\text{context}}$) to compute the logit modifier
Δz.

The fused representation is the five-dimensional context feature vector:

$\mathbf{\psi}\  = \ \lbrack\psi ₀,\ \psi ₁,\ \psi ₂,\ \psi ₃,\ \psi ₄\rbrack\ ᵀ$
(10)

whose components are defined in Table 3. Three components (ψ₀, ψ₁, ψ₄)
are populated by MCP tool outputs and two (ψ₂, ψ₃) by piRAG retrieval
scores, so the ablation masks $\mathbf{m}_{\mathbf{\text{MCP}}}$ = \[1,
1, 0, 0, 1\] and $\mathbf{m}_{\mathbf{\text{piRAG}}}$ = \[0, 0, 1, 1,
0\] cleanly isolate the two channels.

Table 3. Context feature definitions.

  **Feature**   **Name**               **Source**   **Range**   **Definition**
  ------------- ---------------------- ------------ ----------- -------------------------------------------------
  ψ₀            Compliance severity    MCP          \[0, 1\]    0.0 compliant, 0.5 warning, 1.0 critical
  ψ₁            Forecast urgency       MCP          \[0, 1\]    low: 0.1, medium: 0.4, high: 0.7, critical: 1.0
  ψ₂            Retrieval confidence   piRAG        \[0, 1\]    Top citation score / 0.8, capped at 1.0
  ψ₃            Regulatory pressure    piRAG        {0, 1}      1.0 if top doc is regulatory with score \> 0.4
  ψ₄            Recovery saturation    MCP          \[0, 1\]    Fraction of recent decisions routed to recovery

The context weight matrix $\mathbf{\Theta}_{\mathbf{\text{context}}}$ ∈
ℝ^3×5^ maps **ψ** to an additive logit modifier, one row per routing
action. The entries are sign-justified from the operational mandate of
each row, verified against the repository, and refined online via the
REINFORCE update of Section 3.9:

$\mathbf{\Theta}_{\mathbf{\text{context}}}$ =

  $$\lbrack\  - 0.80\ \  - 0.60\ \  - 0.15\ \  - 0.30\ \  + 0.25\ \rbrack$$   *← cold chain*
  --------------------------------------------------------------------------- --------------------------
  $$\lbrack\  + 0.50\ \  + 0.40\ \  + 0.20\ \  + 0.25\ \  + 0.10\ \rbrack$$   *← local redistribution*
  $$\lbrack\  + 0.30\ \  + 0.20\ \  - 0.05\ \  + 0.05\ \  - 0.35\ \rbrack$$   *← recovery*

The cold-chain row is strongly disfavored by compliance violations and
forecast urgency and slightly favored by recovery saturation (because an
overfull recovery pathway argues for keeping produce in cold chain). The
redistribution row is favored by every feature. The recovery row is
strongly disfavored by recovery saturation (capacity constraint) with
moderate positive signals elsewhere. The context-to-logit transformation
is

$\mathbf{\text{Δz}}\  = \ \mathbf{\Theta}_{\mathbf{\text{context}}}\  \cdot \ \mathbf{\psi}$
(11)

clamped element-wise to \[−1, +1\] (approximately half the maximum
physics-driven logit contribution of −2.00 on cold-chain spoilage,
preventing any combination of context features from dominating the base
policy) and then scaled by $\tau_{\text{mod}}$ from Section 3.7.

Every integrated-mode decision produces a five-paragraph causal
explanation: (i) a causal narrative identifying the tool outputs and
retrieval passages that drove the action; (ii) feature-level attribution
of each context feature's logit contribution; (iii) a WITHOUT
counterfactual showing the action that would have been selected with
**ψ** = **0**; (iv) inline knowledge-base citations with extracted
keywords; (v) a provenance hash assembled into a Merkle tree anchored
on-chain (Section 3.15). Because the same **ψ** and
$\mathbf{\Theta}_{\mathbf{\text{context}}}$ drive both the policy and
the explanation, there is no interpretive gap between decided action and
reported reason.

The context vector and its logit modifier are now fully specified; the
routing policy consumes them alongside the state features.

## Regime-Aware Contextual Softmax Policy

The preceding subsections defined every input to the policy: the
physical state vector **φ**(*s*) from perception, the context vector
**ψ** from communication, the regime indicator *τ* from the Bollinger
detector, and the role-specific biases **b~role~**. The routing policy
combines them through a linear logit with context modifier: *\
*$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ z\  = \ \Theta\  \cdot \ \varphi(s)\  + \ \gamma\  \cdot \ \tau\  + \ bonus(m,\ \rho)\  + \ b_{\text{role}}\  + \ \Delta z\  + \ z_{\text{boost}}$
(12)

where each term adds a vector in ℝ³ (one logit per action). Action
probabilities follow the numerically stable softmax: *\
*$\pi(a\ |\ s)\  = \ \frac{exp(z_{a}\  - \ max(z))}{\sum_{i}^{}{exp(z_{i} - \ max(z)}}$
(13)

The state feature vector **φ**(*s*) ∈ ℝ⁶ encodes freshness, inventory
pressure, demand signal, thermal stress, spoilage urgency, and an
interaction term:

$\varphi\  = \ \begin{bmatrix}
1\  - \ \rho \\
min(inv/15000,\ 1) \\
min(ŷ/20,\ 1) \\
clamp((T\  - \ 4)/20,\ 0,\ 1) \\
\rho \\
\rho\  \cdot \ min(inv/15000,\ 1) \\
\end{bmatrix}$ (14)

The base policy weight matrix $\mathbf{\Theta}\ $∈ ℝ^3×6^ is
sign-justified and calibrated so that the base policy produces
approximately 45% cold chain, 45% local redistribution, and 10% recovery
at baseline conditions:

$\mathbf{\Theta}\ $=

  $$\lbrack\  + 0.50\ \  - 0.30\ \  + 0.40\ \  - 0.50\ \  - 2.00\ \  - 1.00\ \rbrack$$   *← cold chain*
  -------------------------------------------------------------------------------------- --------------------------
  $$\lbrack\ \ 0.00\ \  + 0.50\ \  - 0.20\ \  + 0.50\ \  + 2.00\ \  + 1.50\ \rbrack$$    *← local redistribution*
  $$\lbrack\  - 0.50\ \  - 0.30\ \  - 0.20\ \  + 0.30\ \  + 1.50\ \  - 0.30\ \rbrack$$   *← recovery*

Cold chain is favored by freshness (+0.50) and demand (+0.40) and
strongly disfavored by spoilage (−2.00); redistribution is favored by
spoilage (+2.00), inventory pressure (+0.50), and thermal stress
(+0.50); recovery is favored by spoilage with moderate inventory and
thermal components. The governance override is a deterministic rule that
fires when the logit structure signals crisis: if
$z₀\  < \  - 2.0\ \text{and}\ z₁\  > \ z₀\  + \ 3.0$ (a 3.0-unit logit
gap corresponds to \> 95% softmax probability mass on redistribution),
the policy forces *a* = 1 (redistribute), bypassing stochastic sampling.
The thresholds are structurally unreachable from any single context
feature (maximum single-feature contribution to $z₀$ is −0.80); the
override therefore requires simultaneous agreement from the compliance
tool, the spoilage forecaster, and the regulatory retrieval pipeline.

During the simulated cyber outage (processor offline from hour 24
onward), the softmax policy is replaced by a stochastic rerouting model
with action-specific success probabilities (cold chain 0.55, local
redistribution 0.82, recovery 0.60), estimated as the fraction of each
route's operational steps that remain feasible without full telemetry,
reflecting the difficulty of executing each action without full
telemetry.

The context weight matrix is adapted online via a sign constrained
REINFORCE update,

$\mathbf{\Theta}_{\mathbf{\text{context}}}\  \leftarrow \ \mathbf{\Theta}_{\mathbf{\text{context}}}\  + \ \eta_{\text{ctx}}\  \cdot \ N^{- 1}\sum\mathbf{(}\mathbf{e}_{\mathbf{a}}\mathbf{\  - \ \pi)}\mathbf{\psi}^{\mathbf{T}}(R\  - \ R\bar{})$
(15)

with learning rate $\eta_{\text{ctx}}$ = 0.003 (selected by grid search
for validation ARI stability), gradient clip \[−0.5, +0.5\], and
exponentially decaying baseline *R̄* (decay 0.95). The outer product
($\mathbf{e}_{\mathbf{a}}\mathbf{\  - \ \pi}$)
$\mathbf{\psi}^{\mathbf{T}}$ is the REINFORCE policy-gradient direction
in matrix form; the advantage (*R* − *R̄*) applies the reward signal. Two
constraints distinguish this update from standard REINFORCE. A sign mask
freezes the sign of every element of $\Theta_{\text{context}}$ at its
domain-justified value, so that a negative weight by construction (e.g.,
the −0.80 coefficient on compliance for cold chain) cannot cross zero
through stochastic gradient noise. A magnitude bound caps each entry at
its initial absolute value, preventing runaway updates. Together these
constraints permit online refinement of coefficient magnitudes while
ruling out domain-violating weight flips; Section 3.14 evaluates the
convergence behavior empirically. The per-timestep decision loop is
summarized in Algorithm 1.

  **Algorithm 1: Per-Timestep Decision Loop**
  ----------------------------------------------------------------------------------------------------
  Input: sensor telemetry (T, RH, inv, demand), agent role, mode, seed
  1\. Compute spoilage: ρ(t) ← 1 − C(t) via Arrhenius-Baranyi + PINN
  2\. Forecast demand: ŷ ← LSTM (lookback=48); supply: ŝ ← HoltWinters (α=0.5, β=0.2)
  3\. Detect regime: τ ← 1 if \|z-score (demand, w=20)\| \> 2.0 else 0
  4\. Build state vector: φ(s) = \[1−ρ, inv/15000, ŷ/20, thermal, ρ, ρ×$\text{inv}_{\text{press}}$\]
  5\. IF mode ∈ {agribrain, mcp_only, pirag_only }:
  5a. Invoke role-specific MCP tools (Table 2) → tool outputs
  5b. Retrieve piRAG passages (k=4, BM25+TF-IDF) → retrieval scores
  5c. Extract context features: ψ = \[ψ₀, ψ₁, ψ₂, ψ₃, ψ₄\]
  5d. Apply feature mask (mcp_only: \[1, 1, 0, 0, 1\]; pirag_only: \[0, 0, 1, 1, 0\])
  5e. Compute Δz = $\Theta_{\text{context}}$· ψ, clamp to \[−1, +1\]
  5f. Modulate by temporal continuity: Δz ← Δz · $\tau_{\text{mod}}$
  ELSE: Δz ← \[0, 0, 0\]
  6\. Compute logits: z = Θ · φ + γ τ + bonus (m, ρ) + $b_{\text{role}}$ + Δz + $z_{\text{boost}}$
  7\. Governance override: IF $z₀\  < \  - 2.0\ AND\ z₁\  > \ z₀\  + \ 3.0$ THEN force a = 1
  8\. ELSE: sample a \~ π (a \| s) via softmax(z)
  9\. Execute action a; compute waste, carbon, SLCA, reward R = S − 0.50·w
  10\. Log decision to blockchain; generate causal explanation if integrated mode
  11\. Update $\Theta_{\text{context}}\ $via sign constrained REINFORCE
  Output: action a, reward R, explanation trace

Table 4. Operating modes, ablation conditions, and hypothesis mapping.

  **Mode**     **Context**      **PINN**   **SLCA**   **Tests**
  ------------ ---------------- ---------- ---------- ----------------------
  static       None             No         No         Baseline
  hybrid_rl    None             No         No         Component quality
  no_pinn      None             No         Half       Component quality
  no_slca      None             Yes        No         Component quality
  no_context   Disabled         Yes        Yes        H1 (control)
  mcp_only     MCP features     Yes        Yes        H3 (MCP isolation)
  pirag_only   piRAG features   Yes        Yes        H3 (piRAG isolation)
  agribrain    Full             Yes        Yes        H1, H3 (treatment)

With the action selected, the evaluation models translate it into
measurable outcomes.

## Evaluation Models

Three quantities must be computed at each timestep: operational waste,
carbon emissions, and the SLCA composite. Operational waste follows a
calibrated Arrhenius-power form,\
$w_{\text{net}}\  = \ (k_{\text{inst}}\  \cdot \ 10.30)^{0.734}\  \cdot \ (1\  + \ 0.25r_{\text{surplus}})\  \cdot \ (1\  - \ s_{f} \cdot s_{\text{cap}})$
(16)

with $w_{\text{net}}$ = 10.30 and *α* = 0.734, calibrated by non-linear
least squares to historical cooperative waste-rate data so that baseline
waste falls within the FAO 2 to 15% range for fresh produce (FAO, 2019),
yielding *w* ≈ 7.7 % at baseline static conditions (*T* ≈ 4 °C, *k* ≈
0.00274 h⁻¹). A compliance-conditional penalty from MCP data adjusts the
save-capacity factor $s_{\text{cap}}$: 0.70 under critical violation
(mandated 30% disposal), 0.85 under warning (quarantine loss), and 1.05
for redistribution under any violation (an awareness bonus for proactive
rerouting). This coupling is what lets ψ₀ influence waste even when the
physics model is unchanged. Carbon is
$E\  = \ d\  \cdot \ 0.12\  \cdot \ \left( 1\  + \ 0.40\  \cdot \ \theta \right)\text{kg\ }\text{CO}_{2} - eq$
(EPA, 2021), where *d* is route distance and the coefficient 0.40
captures refrigeration COP degradation under thermal stress (Tassou et
al., 2010). SLCA is a weighted composite of community, labor,
resilience, and price-transparency sub-indicators,

$S\  = \ 0.30\ C\  + \ 0.20\ L\  + \ 0.25\ R\  + \ 0.25\ P$ (17)

with weights reflecting cooperative stakeholder priorities (community
resilience weighted highest at 0.30 given the cooperative's social
mandate) and multiplicative attenuation
$q\  = \ 1\ /\ (1\  + \ 0.25\ \theta\  + \ 0.08\ r)$ (UNEP/SETAC, 2020)
giving the attenuated score *S*~att~ = *q* · *S*. The per-step reward is
$R\  = \ S_{\text{att}}\  - \ 0.50\  \cdot \ w_{\text{net}}$, where the
waste-penalty coefficient 0.50 was selected by grid search to maximize
ARI while preserving the relative ordering of waste and SLCA objectives.

## Adaptive Resilience Index and Computational Footprint

A supply chain that minimizes waste while ignoring social performance is
not resilient. The **Adaptive Resilience Index** multiplicatively
combines efficiency, social, and quality factors,

$ARI\  = \ (1\  - \ w_{\text{net}})\  \cdot \ S_{\text{att}}\  \cdot \ (1\  - \ \rho)$
(18)

each factor in \[0, 1\] producing ARI ∈ \[0, 1\]; the multiplicative
structure ensures that poor performance on any dimension suppresses the
overall score (Christopher & Peck, 2004; Pettit et al., 2013). Secondary
endpoints include Reverse Logistics Efficiency
($RLE\  = \ N_{\text{routed}}\ /\ N_{at - risk}$, where at-risk batches
have *ρ* \> 0.10), distributional equity
($Equity\  = \ 1\  - \ \sigma(S_{1},\ \ldots,\ S_{N})$), waste, SLCA,
total carbon, decision latency, and constraint-violation rate.

Efficiency gains from AI routing are undermined if the computational
cost of the models shifts environmental burdens upstream. AGRI-BRAIN
therefore tracks per-step computational footprint transparently: 50 mJ
energy (Strubell et al., 2019), for lightweight CPU inference and
$1.8\  \times \ 10⁻⁶$ L water (Li et al., 2023), integrated cumulatively
per episode and reported alongside operational metrics (Schwartz et al.,
2020). The evaluation framework is complete; the remaining subsections
specify the variation and audit machinery that surrounds every episode.

## Stochastic Perturbation Layer

Deterministic evaluation on a single trajectory cannot distinguish
genuine architectural effects from artefacts of a particular
realization. Seven independent uncertainty sources (Table 5) inject
realistic variation into every evaluation episode, ensuring that
statistical conclusions reflect distributional properties rather than
point estimates.

Table 5. Seven uncertainty sources.

  **Source**                   **Parameter**                               **Default**      **Scope**
  ---------------------------- ------------------------------------------- ---------------- --------------
  Sensor noise                 Temp ±1.5 °C, RH ±5%                        Gaussian         Per timestep
  Demand variability           CV = 18%                                    Multiplicative   Per timestep
  Inventory uncertainty        CV = 15%                                    Multiplicative   Per timestep
  Transport jitter             CV = 15%                                    Multiplicative   Per timestep
  Spoilage model error         $k_{\text{ref}}$ CV 15%, $E_{a}$/R CV 10%   Per episode      Per episode
  Scenario onset jitter        ±4 h                                        Uniform          Per scenario
  Policy weight perturbation   σ = 0.03 on Θ                               Additive         Per seed

Evaluation runs in two modes: **deterministic** (all perturbations
disabled) for exact reproducibility and regression testing, and
**stochastic** (default) for the genuine variation required for
statistical inference. Both modes use seeded random number generators so
that every result is reproducible given its seed.

## Experimental Design

The experimental design combines five scenarios, eight modes, and 20
seeds into $5\  \times \ 8\  \times \ 20\  = \ 800$ episodes that
together isolate the effect of inter-agent communication from model
quality and stochastic variation. The five scenarios (Table 6) span
thermal stress, supply surplus, infrastructure failure, market
volatility, and baseline control. The eight modes (Table 4) progress
from no-intelligence (static: always cold chain) through partial systems
(hybrid RL, no PINN, no SLCA) to full ablation variants (no-context,
MCP-only, piRAG-only) and the full system.

Table 6. Disruption scenarios and stress parameters.

  **Scenario**       **Disruption type**      **Mechanism**                                   **Timing**                   **Key parameters**                **What it tests**
  ------------------ ------------------------ ----------------------------------------------- ---------------------------- --------------------------------- ------------------------------------------------------------------------------------------
  Heatwave           Thermal                  Temperature ramp with exponential tail          Hours 24 to 48, tail to 72   +20 °C peak excursion             Spoilage acceleration, urgency propagation through MCP, governance override activation
  Overproduction     Supply surplus           Inventory surge                                 Hours 12 to 60               2.5× baseline inventory           Redistribution routing, recovery capacity signaling, SLCA reward under surplus
  Cyber outage       Infrastructure failure   Demand collapse and refrigeration degradation   Hour 24 onward               Demand to 15%, +10 °C drift       Communication robustness, MCP fallback, stochastic rerouting under partial observability
  Adaptive pricing   Market volatility        Demand oscillation                              Continuous                   Amplitude 45, period 60, σ = 14   Regime detection, policy stability under non-stationary demand
  Baseline           None                     Unperturbed operation                           Full 72 hours                Standard telemetry                Control condition for all comparisons

The **shared-seed ablation design** is the primary mechanism for
isolating interoperability effects. The four context-ablation modes
(no-context, MCP-only, piRAG-only, full system) share a derived seed per
(scenario, seed) pair, so that identical stochastic perturbations,
demand realizations, and policy noise apply across all four; any ARI
difference between them is causally attributable to the context modifier
**Δz**. The four base modes (static, hybrid RL, no PINN, no SLCA) use
independent seeds because they differ in logit structure and model
availability and cannot be paired meaningfully. Algorithm 2 formalizes
the protocol.

  **Algorithm 2: Shared-Seed Ablation Protocol**
  -----------------------------------------------------------------------------
  FOR each scenario s ∈ {heatwave, overproduction, cyber, pricing, baseline}:
  FOR each seed k ∈ {1, \..., 20}:
  derived_seed ← hash (s, k)
  FOR each mode m ∈ {no_context, mcp_only, pirag_only, agribrain}:
  Initialize RNG with derived_seed (same for all four modes)
  Run episode (s, m, derived_seed) → ARI, waste, SLCA, \...
  FOR each mode m ∈ {static, hybrid_rl, no_pinn, no_slca}:
  Initialize RNG with independent seed (s, m, k)
  Run episode (s, m, independent_seed) → ARI, waste, SLCA, \...
  Paired tests: agribrain vs. no_context within each (s, k) pair

The experimental design produces 800 episodes of paired observations;
the statistical plan specifies how significance is assessed.

## Statistical Analysis Plan

The 800 paired episodes are evaluated under a pre-registered plan that
controls for multiple comparisons, quantifies effect sizes, and
distinguishes statistical from practical significance. The primary
comparison (full system vs. no-context within each scenario) uses paired
permutation tests with 10,000 permutations. Effect sizes are reported as
Cohen's $d_{z}\ $for paired designs, $d_{z}\  = \ \overline{d}/sd$.
Uncertainty is quantified by bias-corrected accelerated bootstrap 95 %
confidence intervals over 10,000 resamples. Multiplicity is controlled
via Holm-Bonferroni correction across the five scenario comparisons.
Pre-registered practical-significance thresholds are
$\Delta ARI\  \geq \ 0.005$, $\Delta Waste\  \leq \  - 0.002$, and
$\Delta SLCA\  \geq \ 0.005$. A power analysis, given 20 seeds and the
seven stochastic perturbation sources of Table 5, supports detection of
$d_{z}$ ≥ 0.7 at *α* = 0.05.

## Blockchain Governance Layer

Distributed decision authority creates a governance risk: without
enforceable rules, autonomous agents may deviate from collectively
agreed policies. Programmable smart contracts provide the enforcement
mechanism, rendering governance rules tamper-proof, auditable, and
executable without manual intervention. Six Solidity smart contracts
(Solidity ≥ 0.8.28) are deployed on a permissioned Ethereum Virtual
Machine (EVM) testnet: **agent registry** (authentication and role
assignment), **decision logger** (immutable routing records), **policy
store** (versioned, tamper-proof parameters including **Θ** and
$\mathbf{\Theta}_{\mathbf{\text{context}}}$), **incentive contract**
(rewards proportional to SLCA performance), **governance contract**
(propose → vote → finalize → queue → execute sequence), and **provenance
registry** (Merkle roots of the explanation evidence chain described in
Section 3.8). Policy updates, including modifications to
$\mathbf{\Theta}_{\mathbf{\text{context}}}$ entries or SLCA weights,
require quorum approval before propagating to operational parameters.
Every episode anchors Merkle roots to the provenance registry, providing
a tamper-evident audit trail that a cooperative board or external
regulator can verify independently. This governance layer closes the
loop opened by the causal explanations of Section 3.8: the same evidence
that drives each decision is cryptographically anchored for downstream
audit.

# Implementation and Case Studies

The evaluation tests the three falsifiable hypotheses declared in
Section 3.2. The scenario analyses in 4.2 through 4.5 trace the
decision-step mechanism under thermal, supply, infrastructure, and
market stress. Section 4.6 addresses H1 (integration superiority)
statistically; 4.8 addresses H3 (component complementarity) through
ablation; 4.11 addresses H2 (communication robustness) through fault
injection. The interoperability check (4.9), provenance audit (4.10),
and cross-scenario synthesis (4.12) establish that the performance gains
are architecturally grounded, institutionally auditable, and
computationally cheap.

## Experimental Logic

The integrated system and the isolated control share identical spoilage
estimation, demand forecasting, social scoring, base policy weights, and
random number streams (Algorithm 2, Section 3.14). The isolated control
routes from the physical state vector φ(s) alone: six features derived
from temperature, inventory, demand, and spoilage risk. It is blind to
whether the current temperature constitutes a regulatory violation,
whether the knowledge base recommends emergency redistribution, what
previous agents decided about recovery routing, or what operational
patterns have been synthesized from recent decisions. The integrated
system opens three communication channels (Figure 3, Section 3.8) that
inject five context features ψ₀ through ψ₄ into the routing logits
through the context weight matrix $\Theta_{\text{context}}$.

Channel 1 delivers agent-to-agent coordination through five typed
message classes: spoilage alerts, surplus notifications, capacity
updates, reroute requests, and acknowledgments. Channel 2, the MCP
protocol layer, produces two kinds of information: external tool outputs
(compliance severity ψ₀, forecast urgency ψ₁) and blockchain queries
that return the fraction of recent decisions routed to recovery (ψ₄).
Channel 3, the piRAG retrieval pipeline, produces retrieval confidence
ψ₂ from the static knowledge base and regulatory pressure ψ₃ when a
regulatory document is the top-ranked passage, with the base
continuously augmented by synthesized operational knowledge on a
six-hour cycle.

The four scenarios prove different structural limits of the
architecture. The heatwave tests performance when the physical signal is
already strong. Overproduction tests performance when the physical state
vector cannot represent coupling between agents. The cyber outage tests
performance when the communication channels themselves degrade. Adaptive
pricing tests performance when no physical constraint has been breached.

## Thermal Stress: Amplification of Physical Evidence into Decisive Action

A +20 °C temperature ramp during hours 24 to 48 drives the Arrhenius
decay rate more than an order of magnitude above baseline. The spoilage
trajectory passes through three phases: during hours 0 to 20, *ρ*
remains near zero while the Baranyi lag-phase adjustment suppresses
effective decay as microbial populations adapt; once the ramp engages,
the Arrhenius rate constant rises exponentially and *ρ* crosses the RLE
threshold of 0.10 near hour 30; after hour 48, *ρ* continues rising
because first-order degradation is irreversible (Figure 4b). Both
systems observe this trajectory identically. They differ in what each
knows it means.

![](media/image5.png){width="6.300694444444445in"
height="4.473611111111111in"}

Figure 4. Heatwave scenario. (a) Temperature and humidity with the
heatwave window shaded. (b) Spoilage risk ρ(t) with the RLE threshold (ρ
= 0.10). (c) Action probability distribution. (d) Rolling-average reward
for static, base RL, and integrated.

Consider hour 28, when cold-chain temperature reaches approximately 14
°C. The isolated control computes thermal stress contribution −0.25 to
the cold-chain logit and +0.25 to redistribution, with a modeled
spoilage urgency of 0.15 adding −0.30 and +0.30. The total cold-chain
logit reaches roughly −1.1, above the −2.0 override threshold (Algorithm
1, line 7). The isolated control tilts toward redistribution but does
not force it.

The integrated system, at the same timestep, queries its MCP tools. The
compliance checker detects 14 °C exceeds the 8 °C FDA threshold and
returns a critical violation (ψ₀ = 1.0), contributing −0.80 and +0.50
through $\Theta_{\text{context}}$. The spoilage forecast tool returns
critical urgency (ψ₁ = 1.0), adding another −0.60 and +0.40. The piRAG
pipeline, triggered by *T* \> 10 °C, retrieves the heatwave contingency
document (ψ₃ = 1.0), adding a further 0.55 points. Combined with the
physics-driven cold-chain logit of −1.1, the cumulative context shift
places the cold chain below −2.0 while lifting redistribution more than
three points above it. The governance override fires.

The mechanism is structural. The physical model and the base policy
produce identical physics-only logits in both systems. The integrated
system acts decisively because three independent institutional signals
converge through a standardized interface onto a common representation,
and no single feature could push the logit past the override boundary:
the maximum single-feature contribution to the cold-chain logit is
−0.80, well above −2.0. Multi-signal convergence is what registers as
action, and this convergence is what isolated reinforcement learning
cannot produce, because the concept of "regulatory violation" is
external to any physical state vector.

The override fires repeatedly across the heatwave episode, each
activation marking a timestep where three institutional signals agreed
on crisis. Waste reduction of −0.007 (Table 8) traces to the
compliance-conditional save factor, which is active only while the
system knows it is operating under violation; without ψ₀, the save
factor defaults to neutral and the policy cannot discriminate between
rescuable and unrescuable produce under thermal stress. Carbon savings
of 192 kg per episode (≈58% of the static baseline, Figure 10) emerge as
a second-order consequence: the policy pursues compliance and social
appropriateness, and the 45 km redistribution route happens to be
shorter than the 120 km cold chain. A system that optimizes carbon
directly can be gamed by rerouting to nearer but unsuitable
destinations; a system that routes by institutional constraint and
incidentally shortens its paths cannot. The composite ARI of 0.615
(Table 7) is the summary; the logit-level boundary crossing is the
mechanism.

## Physical Surplus: Tool-Mediated Feedback Against Pathway Fixation

The second scenario probes a limit the first cannot: decision-relevant
information that is not temperature but coupling between agents. When
inventory exceeds cold-room capacity, overcrowding, compressor strain,
and accelerated spoilage form a feedback loop that raises the effective
Arrhenius rate before ambient temperature changes. The isolated control
detects the surplus through its inventory-pressure feature but cannot
assess pathway saturation. The integrated system can, because the MCP
blockchain chain query returns recent recovery-routing fraction as ψ₄,
and $\Theta_{\text{context}}$ \[2, 4\] = −0.35 disfavors further
recovery routing when the pathway is loaded. The integrated system
consequently sustains full reverse-logistics efficiency across the
surplus window, holding RLE at 1.0 across the surplus window (20-seed
mean 0.978; Table 11), while the Hybrid RL baseline oscillates between
0.7 and 1.0 as it commits and uncommits (Figure 5c). The difference is
not that the integrated system acts earlier; it is that it does not
reverse its own decisions, because it can see them.

The PINN spoilage model and the MCP forecast tool are coupled stages in
a causal chain, not independent contributions. At hour 25, the ODE-only
baseline predicts *ρ* ≈ 0.12, which the forecast tool classifies as high
urgency (ψ₁ = 0.7). The PINN residual correction, responding to
compressor-strain temperature creep, adjusts the estimate to *ρ* ≈ 0.15,
crossing the critical classification threshold; ψ₁ becomes 1.0. Through
*Θ*~context~, the 0.30 increment in ψ₁ produces a 0.30-point additional
logit swing. A 0.03 improvement in spoilage estimation, invisible to any
policy reading *ρ* directly, changes the urgency classification that MCP
consumes, which changes the context modifier, which changes the routing
decision. This neural correction is surfaced to operators through the
quality monitoring panel (Figure 13b). The ablation confirms the
coupling: removing PINN reduces baseline ARI from 0.719 to 0.686 (Table
9), a drop that reflects not only degraded spoilage estimation but
downstream loss of accurate urgency classifications in the MCP pipeline.

![](media/image6.png){width="6.300694444444445in"
height="4.473611111111111in"}

Figure 5. Overproduction scenario. (a) Inventory and demand. (b) Rolling
waste rate. (c) Rolling RLE. (d) SLCA component scores.

The composite index of 0.632 (Table 7) is approximately 80% above the
static baseline of 0.351, but the headline understates the finding.
Under surplus the failure mode shifts from thermal to informational, and
a single-agent physical state vector has no mechanism to represent
pathway saturation. The SLCA improvement of +0.027 and the RLE gain of
+0.033 (Table 8) measure not fewer physical violations but less wasteful
allocation of the recovery capacity the cooperative already owns.

## Channel Breach: Graceful Degradation Under Infrastructure Failure

The third scenario stresses the communication architecture itself. From
hour 24, demand collapses to 15% of nominal and refrigeration degrades
by +10 °C, with a stochastic rerouting model partially replacing the
softmax policy. The integrated system does not collapse to the base
policy floor: it achieves a rerouting success probability of 0.82
against 0.55 for the base policy, and its ARI trajectory decays
gradually rather than discontinuously. Two residual information sources
explain the persistence. MCP tool outputs from the pre-outage period
produce compliance and urgency assessments that remain in the temporal
context window (20 entries, six-hour horizon), and the piRAG-retrieved
cyber outage contingency document persists through the same window. The
integrated system acts on institutional knowledge for several hours
after the channels carrying fresh knowledge have been compromised.

![](media/image7.png){width="6.300694444444445in"
height="1.9604166666666667in"}

Figure 6. Cyber outage. (a) Demand collapse and refrigeration
degradation from hour 24. (b) ARI trajectories. (c) Decision confidence
(1 minus normalized softmax entropy).

The policy confidence trace (Figure 6c) reports this regime awareness.
Before the outage, confidence (one minus normalized softmax entropy)
oscillates between 0.2 and 0.4, reflecting balanced exploration. Once
the outage engages, confidence stabilizes near 1.0: the policy commits
probability mass to redistribution, and because environmental inputs
during sustained outage repeat, entropy collapses at every step. The
flat trace is the behavior expected of a regime-aware contextual policy
that has recognized the shift and committed to a consistent response.

The governance override fires repeatedly during the cyber outage, the
highest-activity scenario across the four tested, because ψ₀ = 1.0, ψ₁ ≥
0.7, and ψ₃ = 1.0 repeatedly drive the cold-chain logit below −2.0.
Under overproduction, adaptive pricing, and baseline, the override never
fires. The contrast is diagnostic. The override is not a frequency-based
safety valve but a semantic one: it engages only when multiple
institutional signals agree on crisis and stays silent otherwise,
regardless of disruption count. The threshold is structurally
unreachable from a single context feature. The cyber outage composite of
0.655 (Table 7) is therefore remarkable less for its numerical value
than for what it implies: performance under compromised channels does
not fall to the base-policy floor because the temporal decay of context
features preserves institutional knowledge for hours beyond channel
failure. The smallest benchmark effect (Cohen's $d_{z}$ = 3.34) appears
here, consistent with the softmax being partially bypassed, yet the
integrated system still wins by +0.019 ARI ($p_{\text{adj}}$ \< 0.001)
through a mechanism that decays rather than fails.

## Pure Information Regime: Knowledge-Grounded Context Without Physical Breach

The fourth scenario is the most demanding test because physical stress
is nearly absent. Demand oscillates with amplitude 45 around a mean of
20 units per step, period 60, with noise σ = 14. No temperature
threshold is crossed, no inventory overflow occurs, no regulatory breach
takes place. MCP tools still report compliance, urgency, and saturation,
but these signals are uniformly benign. What they cannot supply is the
institutional knowledge encoded in operating procedures: how
cooperatives should respond to market disruptions, which redistribution
partners to prioritize, what regulatory documentation is needed for
rapid route changes. The piRAG pipeline surfaces these documents through
scenario-discriminative query expansion, and the retrieval-only
condition outperforms the tool-only condition (0.717 versus 0.716
deterministic ARI, Table 9; both above isolated 0.700). The gap is small
but the direction is consistent: knowledge-grounded context supplies
value that structured tool outputs alone cannot.

![](media/image8.png){width="6.300694444444445in"
height="4.473611111111111in"}

Figure 7. Adaptive pricing. (a) Oscillating demand. (b) Inventory. (c)
Action distribution. (d) SLCA.

Adaptive pricing yields the highest composite of the four scenarios
(0.734, Table 7) and the largest standardized effect ($d_{z}$ = 7.40),
despite being the mildest thermally. The paradox resolves on inspection:
the threat is not heat or surplus but decision coherence under
oscillating signals, and decision coherence is precisely what
institutional knowledge supplies. When demand fluctuates on a 60-step
period with noise, the physical state vector offers only local evidence
of the current oscillation phase, insufficient to decide between holding
inventory and accelerating redistribution. piRAG retrieval surfaces
operational doctrine that resolves this ambiguity by institutional
precedent, and the routing distribution consequently holds a stable
redistribution fraction rather than chasing the oscillation. The
scenario establishes a value proposition orthogonal to thermal
management: the communication architecture supplies decision-relevant
information even when no physical constraint has been breached,
extending its utility to the majority of operating time that a real cold
chain spends outside crisis.

## Quantitative Impact: Testing Integration Superiority

Evaluation is conducted on 20 seeds per scenario-mode combination using
paired permutation tests (Algorithm 2, Section 3.13). Because the
integrated system and the isolated control share identical random
streams within each seed, the only difference between them is the
presence or absence of the three communication channels, and any
performance gap attributes cleanly to the communication architecture
rather than to sampling variance.

ARI improvements range from +0.019 to +0.035 with adjusted *p* \< 0.001
and Cohen's $d_{z}$ between 3.34 and 7.40, all exceeding the
pre-registered practical-significance threshold of ΔARI ≥ 0.005 (Table
7). H1 (integration superiority) is supported across the full design:
positive, statistically significant, and practically meaningful.

The cross-scenario pattern of effect sizes is itself informative. The
largest standardized improvement appears in adaptive pricing and
baseline, where no single environmental disturbance dominates the
physical state vector and the context modifier therefore carries a
proportionally larger share of the decision. The smallest improvement
appears under cyber outage, where the stochastic rerouting model
replaces the softmax for most of the episode. Heatwave sits in between:
the Arrhenius signal is already strong enough to shift the softmax
without context, so the marginal value of context is compressed into the
override boundary-crossing of Section 4.2. Context is most valuable
where physical evidence is weakest, not where it is strongest.

Table 7. Primary endpoint: ARI, integrated system versus isolated
control, 20-seed stochastic benchmark.

  **Scenario**       **ΔARI**   **95% CI**         $$\mathbf{p}_{\mathbf{\text{adj}}}$$   **Cohen's** $\mathbf{d}_{\mathbf{z}}$   **\> 0.005?**
  ------------------ ---------- ------------------ -------------------------------------- --------------------------------------- ---------------
  Heatwave           +0.024     \[0.022, 0.027\]   \< 0.001                               3.54                                    Yes
  Overproduction     +0.029     \[0.027, 0.031\]   \< 0.001                               5.89                                    Yes
  Cyber outage       +0.019     \[0.016, 0.021\]   \< 0.001                               3.34                                    Yes
  Adaptive pricing   +0.032     \[0.031, 0.034\]   \< 0.001                               7.40                                    Yes
  Baseline           +0.035     \[0.032, 0.037\]   \< 0.001                               5.56                                    Yes

Pairwise comparisons use 10,000-resample paired permutation tests;
confidence intervals are bias-corrected accelerated bootstrap over 20
seeds; Holm-Bonferroni correction is applied across the five scenarios.

## Mechanistic Decomposition of the Performance Gain

Disaggregating ARI into waste, SLCA, carbon, RLE, and equity identifies
which channel drives each component (Table 8). Waste reduction traces to
the compliance-conditional save factor: ψ₀ enables a
compliance-dependent penalty on cold-chain save under critical
violations that the isolated control cannot apply, because the concept
of violation is external to its state space. The SLCA improvement is
largest in baseline (+0.034), precisely where the base policy routes
least toward redistribution by default; because redistribution outscores
cold chain on all four social sub-indicators, even a small routing shift
reshapes the episode-level social profile substantially.

The baseline RLE improvement (+0.056) is the single most informative
cell in Table 8. Baseline conditions offer no physical trigger for
redistribution, so the base policy defaults to cold chain. The piRAG
pipeline, by contrast, surfaces SOP guidance recommending proactive
redistribution at elevated spoilage risk, identifying at-risk batches
the urgency feature alone treats as acceptable. The gain measures the
value of institutional foresight: the system acts on guidance with no
physical counterpart in the state vector, which is exactly the retrieval
channel's design purpose. Carbon reductions follow as a structural
consequence of the routing shift, and equity improvements (+0.005 to
+0.026) reflect the load-balancing role of ψ₄, which prevents any single
downstream partner from absorbing a disproportionate share of the
redirected flow.

Table 8. Secondary metrics: integrated minus isolated control, 20-seed
means.

  **Scenario**       **Waste**   **SLCA**   **Carbon (kg)**   **RLE**   **Equity**
  ------------------ ----------- ---------- ----------------- --------- ------------
  Heatwave           −0.007      +0.022     −192              +0.013    +0.016
  Overproduction     −0.008      +0.027     −239              +0.033    +0.021
  Cyber outage       −0.004      +0.016     −123              0.000     +0.005
  Adaptive pricing   −0.006      +0.030     −240              +0.027    +0.024
  Baseline           −0.006      +0.034     −257              +0.056    +0.026

Negative ΔWaste and ΔCarbon denote improvement; positive ΔSLCA, ΔRLE,
ΔEquity denote improvement.

## Non-Redundant Contributions of MCP and piRAG: Testing Component Complementarity

The aggregate statistics establish that the integrated architecture
outperforms its information-starved counterpart, but not why. Either
channel alone might account for the entire gap, making the second
redundant, or both might contribute identical information, making either
removable. The ablation resolves this by testing whether the two
channels answer different questions and whether absence of either
produces a deficit the other cannot close.

The tool-only mask \[1, 1, 0, 0, 1\] retains ψ₀, ψ₁, ψ₄ and zeroes ψ₂,
ψ₃; the retrieval-only mask \[0, 0, 1, 1, 0\] does the opposite. Across
all five scenarios, both single-channel conditions exceed the isolated
control, and the fully integrated system exceeds the better
single-channel condition by +0.008 to +0.017 ARI with zero rank
inversions (Table 9; Figure 8a shows the first four scenarios). All
three parts of H3 are supported: tool-only \> isolated, retrieval-only
\> isolated, and integrated \> max (tool-only, retrieval-only) in every
scenario.

The mechanism reflects a division of labor. MCP tools answer *what is
happening*: whether the temperature is a violation, how urgent the
spoilage forecast is, how saturated the recovery pathway is. piRAG
documents answer *what should be done about it*: what the contingency
plan recommends, what redistribution protocols apply. The tool-only
condition captures operational state but not institutional response when
ψ₀ = 1.0 and ψ₃ = 0; the retrieval-only condition captures institutional
guidance but not real-time severity when ψ₃ = 1.0 and ψ₀ = 0. Only the
integrated configuration captures both jointly. This is the
architectural distinction between AGRI-BRAIN and isolated reinforcement
learning: standardized communication channels make heterogeneous
information sources composable at runtime, removing the need to
pre-encode every possible context signal into a monolithic state
representation.

The ablation also establishes that the SLCA-shaped reward is an active
routing driver, not a reporting metric. Removing SLCA yields the
second-worst ARI across all scenarios (baseline 0.567 versus isolated
0.719, a drop of 0.152; Figure 8a), the largest single-component effect
in the ablation. Without SLCA, the policy optimizes waste alone and
routes nearly everything to the cold chain, which achieves the lowest
raw waste at baseline temperatures. The consequences are visible in the
SLCA sub-indicators: community resilience drops to approximately 0.40,
labor fairness to 0.50, and carbon remains near 4200 kg as the 120 km
route dominates. The redistribution bonus *B*~SLCA~ = \[−0.35, +0.60,
−0.10\] converts ethical objectives from passive metrics into active
incentives, favoring the shorter, community-oriented route whenever
social performance benefits. For a cooperative whose mission includes
food access and community resilience, this is the most practically
consequential architectural feature

![](media/image9.png){width="6.300694444444445in"
height="1.9534722222222223in"}

Figure 8. Eight-mode ablation: ARI, waste, and RLE across all five
scenarios. Zero rank inversions.

Table 9. Full eight-mode ARI, deterministic evaluation.

  **Scenario**       **Static**   **No SLCA**   **Hybrid**   **No PINN**   **Isolated**   **Tool-Only**   **Retrieval**   **Integrated**
  ------------------ ------------ ------------- ------------ ------------- -------------- --------------- --------------- ----------------
  Heatwave           0.343        0.504         0.550        0.566         0.585          0.595           0.599           0.615
  Overproduction     0.351        0.496         0.546        0.571         0.601          0.616           0.615           0.632
  Cyber outage       0.381        0.543         0.549        0.591         0.637          0.641           0.644           0.655
  Adaptive pricing   0.433        0.549         0.619        0.664         0.700          0.716           0.717           0.734
  Baseline           0.448        0.567         0.651        0.686         0.719          0.732           0.745           0.753

Deterministic single-seed evaluation. Zero rank inversions across all 40
cells: Static \< No SLCA \< Hybrid \< No PINN \< Isolated \< Tool-Only ≈
Retrieval \< Integrated holds in every scenario.

## Interoperability Validation: Three Channels, One Representation

The ablation establishes that the two channels carry different
information; this section establishes that they are standardized,
inspectable pathways rather than hard-coded function calls, which is a
precondition for structural explainability.

Channel 1, direct agent-to-agent messaging, propagates information
produced at one lifecycle stage to the next agent before it becomes the
active decision-maker, through the five typed message classes named in
Section 3.5. When the processor detects surplus at hour 20, its typed
surplus notification reaches the distributor, which updates its internal
state to anticipate incoming volume; the anticipatory behavior registers
as sustained RLE across the overproduction window (Figure 5c).
Structured typing ensures downstream agents parse messages
unambiguously.

Channel 2, the MCP protocol layer, captures genuine JSON-RPC 2.0 request
and response pairs per episode through the protocol recorder. Each tool
invocation produces a structured request (method name, typed parameters,
request identifier) and receives a structured response (result object or
error with code and message). The channel serves two functions.
*Agent-to-external-tool*: the compliance checker, spoilage forecaster,
and SLCA lookup provide operational assessments that populate ψ₀ and ψ₁.
*Agent-to-agent-via-blockchain*: the chain query tool retrieves past
routing decisions from the decision logger contract and computes ψ₄, so
the distributor effectively learns what the farm and processor decided
by querying the shared ledger. Role-specific dispatch confirms genuine
interoperability: the distributor invokes five tools while the farm
invokes three (Table 2, Section 3.6). Across the 20-seed benchmark the
protocol records 200 tool calls per episode per scenario with zero
dispatch errors, confirming the pathway is reliable at operational
scale.

![](media/image10.png){width="6.300694444444445in"
height="2.455696631671041in"}

Figure 9. ARI degradation under four fault categories across all
scenarios.

Channel 3, piRAG-mediated retrieval, also serves two functions. *Static
institutional knowledge*: the 20-document base contains pre-loaded
regulatory guidance, SOPs, SLCA reference standards, and contingency
plans that the BM25+TF-IDF hybrid retriever surfaces in response to
physics-informed queries, populating ψ₂ and ψ₃. *Dynamic operational
knowledge*: every 24 timesteps, the feedback mechanism synthesizes past
routing decisions into the base, capturing operational patterns such as
the conditions under which the system routed to redistribution, the SLCA
scores that resulted, and the spoilage outcomes that followed.
Subsequent retrievals surface these synthesized entries alongside the
static documents. The full runtime pipeline, including tool invocation
counts, current context feature values, the knowledge base, and the live
protocol interaction log, is exposed to operators through the MCP/piRAG
panel (Figure 13c).

The three channels converge in ψ = \[ψ₀, ψ₁, ψ₂, ψ₃, ψ₄\]; all five
features enter one *Θ*~context~ that the policy consumes. Heterogeneous
sources (structured tool APIs, blockchain audit trails, unstructured
document retrieval, synthesized operational knowledge) unify into a
common representation that modifies routing at the logit level. The
circuit breaker (failure threshold = 3, exponential backoff) ensures
that failure in any single channel degrades the system gracefully to the
base policy rather than blocking the pipeline. The architectural
contrast with isolated RL is clean: in isolated RL, context must be
encoded into the state representation at training time and new signals
require retraining; in AGRI-BRAIN, new signals are added as a new row of
$\Theta_{\text{context}}$ with a new tool or document category, and the
policy reads them at runtime.

## Causal Traceability and Decision Provenance

Explainability is a structural consequence of the interoperability
design, not a layer added afterwards. The same five context features
that enter the logit modifier appear in every causal explanation, which
contains five components: a BECAUSE narrative identifying the dominant
signals, feature-level attribution for each ψ dimension, a WITHOUT
counterfactual, inline knowledge-base citations classified into
regulatory thresholds, SOPs, waste-hierarchy guidance, and governance
directives, and a SHA-256 provenance chain assembled into a Merkle tree
whose root is anchored on-chain through the provenance registry.

Analysis of the 20-seed benchmark confirms three structural properties:
*causal chain coverage* is 100% (every context-influenced decision
carries all five components), *sign consistency* is 100% (the stated
dominant feature agrees in sign with its $\Theta_{\text{context}}$
entry), and *provenance integrity* is 100% (no Merkle root hash
mismatches). When the explanation states that a critical compliance
violation drove redistribution, ψ₀ = 1.0 and $\Theta_{\text{context}}$
\[1, 0\] = +0.50 are the actual logit-level mechanism rather than a post
hoc rationalization. The engine traces from the final action through the
logit modifier back to the specific tool output or retrieved document,
which is possible only because the context pipeline transforms
heterogeneous MCP outputs and piRAG retrievals into a structured feature
vector with sign-justified weights. These per-decision records, and
their temporal aggregation over an episode, are surfaced to stakeholders
through the causal explanation and decisions timeline panels described
in Section 4.13, supporting audit without access to policy weights.

The architectural implication is that explainability in AGRI-BRAIN is
not achieved by attaching a post-hoc interpreter to an opaque policy but
by designing the policy-to-context interface to be inspectable by
construction. The features that drive the policy are the features that
populate the explanation; the weights that scale them are the weights
that appear in the attribution; the Merkle-rooted chain anchors every
explanation component to tamper-evident cryptographic anchoring.
Post-hoc attribution methods (SHAP, LIME, attention visualization)
produce approximations the user must trust; AGRI-BRAIN produces traces
the user can verify.

## Robustness Under Communication Faults: Testing H2

H2 predicts that the performance gain is preserved under four fault
categories with \|ΔARI\| \< 0.01 per stressor. All four categories
produce \|ΔARI\| well below threshold (Table 10). The worst case is
telemetry delay under baseline at 0.007 absolute ΔARI. Under tool-fault
injection, the circuit breaker opens after three consecutive failures
and the system reverts to the base policy. The architecture is designed
so that communication failure drops performance to the isolated-control
level, never below it; the context improvement is additive, not
load-bearing. H2 (communication robustness) is supported across all four
fault categories and all five scenarios, with maximum \|ΔARI\| = 0.007
\< 0.01.

Table 10. Stress test \|ΔARI\| for the integrated system across fault
categories (absolute values).

  **Stressor**      **Heatwave**   **Overproduction**   **Cyber**   **Pricing**   **Baseline**   **Max \|ΔARI\|**
  ----------------- -------------- -------------------- ----------- ------------- -------------- ------------------
  Sensor noise      0.006          0.001                0.001       0.002         0.007          0.007
  Missing data      \< 0.001       \< 0.001             \< 0.001    0.002         \< 0.001       0.002
  Telemetry delay   0.006          0.003                0.002       0.001         0.007          0.007
  Tool fault        0.004          0.001                0.003       0.002         0.006          0.006

Deltas are \|integrated ARI with stressor − integrated ARI without
stressor\|. All values are below the pre-registered robustness threshold
of 0.01.

## Cross-Scenario Synthesis: Environmental, Social, and Computational Accounts

The scenario studies, the H1 benchmark, the decomposition, the H3
ablation, the interoperability check, the provenance audit, and the H2
fault test converge on a single finding: the communication architecture
delivers consistent, mechanistically traceable, and institutionally
auditable value across every tested regime at a computational cost
negligible against the operational gains.

![](media/image11.png){width="6.300694444444445in"
height="4.513888888888889in"}

Figure 10. Cross-scenario comparison: ARI, RLE, waste, SLCA, and
carbon*.*

Waste falls from 0.077 to 0.126 range of the static policy to 0.019 to
0.033 range of the integrated system (Table 11), a 74% to 81% reduction
driven by the compliance-conditional save factor and the shift of
at-risk batches to redistribution. Carbon emissions fall by 50% to 58%
across all scenarios (Figure 10), driven entirely by the routing shift
from the 120 km cold chain to the 45 km redistribution path: no fleet
electrification, alternative fuels, or infrastructure investment is
required. Because carbon is never an explicit optimization target, the
system cannot be gamed into reducing reported carbon at the expense of
food access or compliance. SLCA rises from 0.46 to 0.51 under the static
policy to 0.73 to 0.81 under the integrated system, registering
simultaneous gains in community resilience, labor fairness, carbon
intensity, and price transparency. The cross-scenario figure renders
these gains simultaneously (Figure 10), confirming that they are not
purchased at one another's expense.

Decision latency under the context-enabled modes ranges from 3.75 to
5.05 ms per timestep (Table 11), representing less than 0.001% of the
15-minute decision cycle. The latency-quality frontier (Figure 12)
positions the integrated system at +0.019 ARI for +4.0 ms relative to
the no-context reference. Computational overhead is an engineering
detail five orders of magnitude below the operational cycle, not a
speed-versus-quality trade-off. The context layer purchases
statistically robust (H1), mechanistically explained (Section 4.7),
institutionally auditable (Section 4.10), and fault-tolerant (H2)
improvements at negligible operational cost.

![](media/image12.png){width="6.300694444444445in"
height="2.6868055555555554in"}

Figure 11. Green AI footprint: cumulative energy and water per episode.

![](media/image13.png){width="6.300694444444445in"
height="2.4270833333333335in"}

Figure 12. Latency-quality Pareto frontier: decision latency vs. ARI.

Table 11. Cross-scenario comparison: static, hybrid RL, and integrated
methods.

  **Scenario**       **Method**   **ARI**   **RLE**   **Waste**   **SLCA**   **Carbon (kg)**   **Latency (ms)**
  ------------------ ------------ --------- --------- ----------- ---------- ----------------- ------------------
  Heatwave           Static       0.343     0.000     0.117       0.461      4864              0.02
  Heatwave           Hybrid RL    0.550     0.950     0.043       0.697      2600              0.07
  Heatwave           Integrated   0.615     0.994     0.022       0.759      2066              5.05
  Overproduction     Integrated   0.632     0.978     0.033       0.725      2022              4.25
  Cyber outage       Integrated   0.655     0.800     0.032       0.739      2363              4.21
  Adaptive pricing   Integrated   0.734     0.903     0.020       0.798      2069              3.95
  Baseline           Static       0.448     0.000     0.077       0.513      4210              0.02
  Baseline           Hybrid RL    0.651     0.649     0.034       0.710      2738              0.07
  Baseline           Integrated   0.753     0.980     0.019       0.812      2004              3.75

Integrated-method rows are 20-seed stochastic means. Static and hybrid
RL rows are reported for heatwave and baseline to anchor the regime
endpoints; intermediate scenarios for those methods follow the same
ordering and are reported in the reproducibility appendix. "Hybrid RL"
denotes the static-to-RL switchover controller (Section 3.9), not a
hybrid of integrated and isolated systems.

## Operational Interface and Deployment Readiness

The framework is implemented as a complete operational platform rather
than a research prototype, and the blockchain governance layer is
functional rather than decorative. Six Solidity smart contracts pass all
Hardhat test cases: the agent registry (role authentication), the
decision logger (immutable routing records), the policy store (versioned
parameters), the governance contract (quorum-based lifecycle), the
provenance registry (Merkle root anchoring), and the SLCA rewards
contract (performance-proportional incentives). Policy parameter
updates, such as modifications to $\Theta_{\text{context}}$ entries or
SLCA weights, require quorum approval through the propose, vote,
finalize, queue, execute sequence before propagating to operational
parameters. Every episode anchors Merkle roots from the provenance
registry, providing a tamper-evident audit trail that a cooperative
board or external regulator can verify independently.

A React-based frontend renders the architectural logic as inspectable
surfaces (Figure 13). The operations dashboard (Figure 13a) consolidates
key performance indicators, live cold-chain telemetry, and regulatory
threshold overlays into a single pane so that staff can detect drift
before it translates into violations. The quality monitoring panel
(Figure 13b) renders the PINN-corrected spoilage trajectory alongside
its ODE-only baseline, exposing the neural residual mechanism analyzed
in Section 4.3; when the two trajectories diverge, the operator sees
immediately that the urgency classification driving the current routing
decision is grounded in the physics-informed correction rather than the
kinetic model alone. The MCP and piRAG panel (Figure 13c) exposes the
three communication channels analyzed in Section 4.9: cumulative tool
invocation counts per tool, the current values of the five context
features ψ₀ through ψ₄, a browser for the 20-document knowledge base and
its dynamically synthesized entries, and a live log of JSON-RPC 2.0
protocol interactions.

The expanded causal explanation panel (Figure 13d) materializes the
provenance audit of Section 4.10 into a single decision record: a
BECAUSE narrative identifying the dominant context signals, a WITHOUT
counterfactual showing the action the policy would have selected without
context, feature-level attribution for each ψ dimension, inline
knowledge-base citations classified by document type, and the Merkle
root anchoring the record on-chain. Because the same five features and
weights drive both the policy and the explanation, there is no
interpretive gap between what the system decided and what the system
shows the operator. The analytics panel (Figure 13e) closes the loop
with a performance summary and cross-scenario validation view that
exposes the underlying ARI, waste, SLCA, carbon, and latency
measurements supporting the claims of this section.

![](media/image14.jpeg){width="6.270138888888889in"
height="4.284722222222222in"}

Figure 13. AGRI-BRAIN operational interface. The React-based frontend
translates the architectural logic into inspectable surfaces. (a) The
operations dashboard overlays live telemetry with regulatory thresholds.
(b) The quality panel exposes the PINN-corrected spoilage trajectory
alongside its ODE-only baseline. (c) The MCP/piRAG panel provides a live
protocol interaction log and context feature tracker. (d) The expanded
causal explanation panel materializes the provenance audit, providing
BECAUSE/WITHOUT reasoning, feature-level attribution, and Merkle-rooted
on-chain verification for every routing decision. (e) The analytics
panel summarizes the cross-scenario performance benchmarks*.*

These interfaces support three deployment phases. *Monitoring*: sensors
and the dashboard operate without routing changes, letting operators
validate predictions against observed outcomes. *Advisory*: decision
memos with operator override let staff accept, reject, or modify each
recommendation. *Autonomous*: direct execution with blockchain logging
supports unattended operation. Each phase delivers value without
requiring full automation commitment, addressing the principal adoption
barrier for AI systems in cooperative agricultural operations.

# Conclusion 

This paper presented AGRI-BRAIN, a protocol-mediated, physics-informed
multi-agent framework for perishable supply-chain decision-making. The
central result is that the performance gap between isolated routing
intelligence and institutionally grounded decision-making is not
resolved by improving prediction alone. It is resolved by enabling
heterogeneous agents, tools, and knowledge sources to exchange context
in a form that directly modifies action selection. Across the spinach
cold-chain case study, AGRI-BRAIN showed that protocol-mediated
communication improves resilience, reduces waste and emissions, and
strengthens social performance relative to otherwise comparable
no-context architectures. These gains arose not from a different
physical backbone alone, but from a structured decision interface
through which compliance status, spoilage urgency, recovery saturation,
and retrieved operational guidance became actionable within the routing
policy itself.

The results also establish that this advantage is architectural rather
than incidental. First, interoperability is realized as a genuine
decision mechanism: peer-agent messages, external tool outputs, and
retrieved institutional knowledge are transformed into a unified
contextual representation that alters routing behavior at the policy
level. Second, explainability is achieved structurally rather than post
hoc, because the same contextual quantities that modify the logits also
generate the causal explanation and provenance trail. Third, the
framework demonstrates a direct physics-to-policy pathway, in which
improvements in the physics-informed spoilage model sharpen urgency
classification and thereby change downstream decisions. Fourth,
sustainability and social accountability function as active routing
drivers rather than passive reporting metrics, allowing redistribution,
resilience, and equity considerations to influence action selection
alongside operational performance. Together, these results show that
AGRI-BRAIN is not simply an integration of digital components, but a
decision architecture for coordinated, explainable, and auditable
autonomy.

Several next steps follow naturally from this work. One is to extend
AGRI-BRAIN from episodic contextual adaptation to more continuous
learning under changing operating regimes, while preserving the
sign-constrained and auditable structure of the policy interface. A
second is to evaluate the explanation and oversight interfaces with
human operators in advisory settings, where trust, usability, and
intervention behavior become part of system performance. A third is to
test the architecture under broader competitive and adversarial
supply-chain conditions, where agents may operate under partially
aligned or conflicting objectives. These directions would strengthen the
path from simulation-based validation to decision-support deployment in
real agri-food networks.

More broadly, the implications of this study extend beyond the spinach
case study. For process systems engineering, the key lesson is that the
frontier of intelligent logistics is not defined solely by the accuracy
of individual predictors, whether spoilage estimators, demand
forecasters, or routing policies. It is increasingly defined by whether
heterogeneous sources of physical, institutional, and operational
knowledge can be communicated, verified, and acted upon within a common
decision architecture. AGRI-BRAIN shows that when such communication is
standardized and auditable, isolated intelligence can be converted into
coordinated, explainable, and accountable supply-chain action. In that
sense, the path forward for autonomous supply chains is not simply
larger models, but richer and more reliable protocols for
decision-making.

*Appendix*

Additional technical details, code snippets, or datasets used in the
study.

# Acknowledgments {#acknowledgments .list-paragraph}

Tentative outline of the write up.

-   Funding sources and support

-   Contributions of collaborators and institutions

# Declaration of Competing Interest {#declaration-of-competing-interest .list-paragraph}

The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to
influence the work reported in this paper.

# CRediT authorship contribution statement {#credit-authorship-contribution-statement .list-paragraph}

Need to update.

# References {#references .list-paragraph}

Baranyi, J., & Roberts, T. A. (1994). A dynamic approach to predicting
bacterial growth in food. International Journal of Food Microbiology*,*
23, 277--294.Barbosa-Póvoa, A. P., da Silva, C., & Carvalho, A. (2018).
Opportunities and challenges in sustainable supply chain: An operations
research perspective. European Journal of Operational Research*,* 268,
399--431.Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot,
A., Tabik, S., Barbado, A., Garcia, S., Gil-Lopez, S., Molina, D.,
Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable Artificial
Intelligence (XAI): Concepts, taxonomies, opportunities and challenges
toward responsible AI. Information Fusion*,* 58, 82--115.Christopher,
M., & Peck, H. (2004). Building the Resilient Supply Chain. The
International Journal of Logistics Management*,* 15, 1--14.EPA. (2021).
Emission Factors for Greenhouse Gas Inventories. In (2021 Edition ed.).
Washington, DC: EPA Center for Corporate Climate Leadership.FAO. (2019).
The State of Food and Agriculture 2019: Moving forward on food loss and
waste reduction: Food and Agriculture Organization of the United Nations
(FAO).Giannakourou, M. C., & Taoukis, P. S. (2003). Kinetic modelling of
vitamin C loss in frozen green vegetables under variable storage
conditions. Food Chemistry*,* 83, 33--41.Grossmann, I. E. (2005).
Enterprise-wide optimization: A new frontier in process systems
engineering. AIChE Journal*,* 51, 1846--1857.Grossmann, I. E. (2012).
Advances in mathematical programming models for enterprise-wide
optimization. Computers & Chemical Engineering*,* 47, 2--18.Kazancoglu,
Y., Ekinci, E., Mangla, S. K., Sezer, M. D., & Kayikci, Y. (2021).
Performance evaluation of reverse logistics in food supply chains in a
circular economy using system dynamics. Business Strategy and the
Environment*,* 30, 71--91.Kazi, M.-K., & Hasan, M. M. F. (2024). Optimal
and secure peer-to-peer carbon emission trading: A game theory informed
framework on blockchain. Computers & Chemical Engineering*,* 180,
108478.Labuza, T. (1982). Theory and application of Arrhenius kinetics
to the prediction of nutrient losses in foods. Food Tech.*,* 36,
55--74.Leitão, P. (2009). Agent-based distributed manufacturing control:
A state-of-the-art survey. Engineering Applications of Artificial
Intelligence*,* 22, 979--991.Li, P., Yang, J., Islam, M. A., & Ren, S.
(2023). Making AI Less \"Thirsty\": Uncovering and Addressing the Secret
Water Footprint of AI Models. arXiv preprint*,* abs/2304.03271.Lund, S.,
Manyika, J., Woetzel, L., Barriball, E., Krishnan, M., Alicke, K.,
Birshan, M., George, K., Smit, S., Swan, D., & Hutzler, K. (2020). Risk,
resilience, and rebalancing in global value chains. In: McKinsey Global
Institute.Pettit, T. J., Croxton, K. L., & Fiksel, J. (2013). Ensuring
Supply Chain Resilience: Development and Implementation of an Assessment
Tool. Journal of Business Logistics*,* 34, 46--76.Pistikopoulos, E. N.,
Barbosa-Povoa, A., Lee, J. H., Misener, R., Mitsos, A., Reklaitis, G.
V., Venkatasubramanian, V., You, F., & Gani, R. (2021). Process systems
engineering - The generation next? Computers & Chemical Engineering*,*
147, 107252.Saberi, S., Kouhizadeh, M., Sarkis, J., & Shen, L. (2019).
Blockchain technology and its relationships to sustainable supply chain
management. International Journal of Production Research*,* 57,
2117--2135.Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020).
Green AI. Communications of the ACM*,* 63, 54--63.Strubell, E., Ganesh,
A., & McCallum, A. (2019). Energy and policy considerations for deep
learning in NLP. In Proceedings of the 57th Annual Meeting of the
Association for Computational Linguistics (pp. 3645--3650): Association
for Computational Linguistics.Swinnen, J., & McDermott, J. (2020).
COVID-19 and global food security. Washington, DC: International Food
Policy Research Institute (IFPRI).Tassou, S. A., Lewis, J. S., Ge, Y.
T., Hadawey, A., & Chaer, I. (2010). A review of emerging technologies
for food refrigeration applications. Applied Thermal Engineering*,* 30,
263--276.Teigiserova, D. A., Hamelin, L., & Thomsen, M. (2020). Towards
transparent valorization of food surplus, waste and loss: Clarifying
definitions, food waste hierarchy, and role in the circular economy.
Science of The Total Environment*,* 706, 136033.UNEP/SETAC. (2020).
Guidelines for Social Life Cycle Assessment of Products and
Organizations 2020. In: United Nations Environment
Programme.Venkatasubramanian, V. (2019). The promise of artificial
intelligence in chemical engineering: Is it here, finally? AIChE
Journal*,* 65, 466--478.Venkatasubramanian, V., & Chakraborty, A.
(2025). Quo Vadis ChatGPT? From large language models to Large Knowledge
Models. Computers & Chemical Engineering*,* 192, 108895.

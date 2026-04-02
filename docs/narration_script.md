# AGRI-BRAIN Video Narration Script

Use this script with ElevenLabs, Google TTS, or any text-to-speech service.
Each section maps to a slide in the video. Paste one section at a time or the full script.

---

## [0:00 - 0:06] Title

Welcome to AGRI-BRAIN — an Adaptive Supply-Chain Intelligence System for Sustainable Food Logistics. This video walks through every component of the system.

## [0:06 - 0:12] The Problem

Fresh produce spoils during transport. Temperature excursions double decay rates. Demand uncertainty creates surplus and shortages. Every batch faces a routing choice: cold chain transport, local redistribution, or recovery. And each decision has social impact on carbon emissions, labor conditions, and community food security.

## [0:12 - 0:18] Solution Overview

AGRI-BRAIN solves this with a 12-phase decision pipeline. Starting from IoT sensors, through physics-informed spoilage prediction, LSTM demand forecasting, multi-agent dispatch, MCP tool invocation, piRAG knowledge retrieval, context feature extraction, policy network computation, all the way to blockchain-anchored provenance. The system evaluates 5 scenarios across 8 operating modes — 40 episodes total.

## [0:18 - 0:25] The 5 Agents

Five supply-chain agents manage the produce lifecycle. The Farm Agent handles the first 18 hours, preserving freshness. The Processor Agent manages hours 18 to 36, ensuring cold chain integrity. The Cooperative Agent overlays hours 12 to 30, coordinating governance and equity. The Distributor Agent handles hours 36 to 54, optimizing logistics. And the Recovery Agent manages end-of-life from hour 54 onward, diverting waste to composting, animal feed, or food banks. Each agent has role-specific biases and communicates via typed messages like spoilage alerts and reroute requests.

## [0:25 - 0:31] Physics-Informed Models

Two predictive models work together. The PINN spoilage model uses an Arrhenius-Baranyi differential equation to compute decay rates based on temperature and humidity. Key parameters include a baseline rate of 0.0021 per hour, activation energy of 8000 Kelvin, and a 12-hour microbial lag phase. The LSTM demand forecaster uses 16 hidden units with Bollinger band regime detection to classify demand as normal, warning, or anomaly.

## [0:31 - 0:38] MCP Protocol

The Model Context Protocol provides 12 tools via JSON-RPC 2.0. These include compliance checking against FDA temperature limits, spoilage forecasting with urgency levels, SLCA weight lookups, blockchain audit trail queries, policy oracle governance checks, footprint tracking, and piRAG knowledge retrieval. Each agent invokes a role-specific subset of these tools during its decision process.

## [0:38 - 0:44] piRAG Pipeline

piRAG — Physics-informed Retrieval-Augmented Generation — maintains a 20-document knowledge base across 6 categories: regulatory guidelines, standard operating procedures, SLCA methodology, environmental standards, technical specifications, and contingency protocols. Retrieval uses BM25 plus TF-IDF hybrid scoring with top-4 selection. A key innovation is physics-informed reranking: the Arrhenius-based rescoring surfaces different documents under different temperature conditions. piRAG consistently contributes more to system performance than MCP alone.

## [0:44 - 0:51] Context Features and Policy

MCP and piRAG outputs are distilled into a 5-dimensional context vector: compliance severity, forecast urgency, retrieval confidence, regulatory pressure, and recovery saturation. This vector is multiplied by a learned weight matrix to produce logit adjustments for each of the three routing actions. The policy network then combines state features, regime tilt, SLCA bonuses, agent-specific biases, and the context modifier through a softmax function to produce action probabilities.

## [0:51 - 0:57] SLCA and Blockchain

Every decision is scored on four social life-cycle pillars: Carbon reduction, Labor fairness, Community Resilience, and Price Transparency. The Adaptive Resilience Index combines waste reduction, SLCA score, and shelf quality into a single metric. For provenance, each MCP tool output and piRAG citation is SHA-256 hashed, combined into a Merkle tree, and the root is anchored on a Hardhat Solidity smart contract for immutable auditability.

## [0:57 - 1:03] Ablation Results

Systematic ablation across 5 scenarios and 8 modes reveals each component's contribution. Removing SLCA causes the largest ARI drop to 0.50-0.58. Removing the PINN model increases waste. The full AGRI-BRAIN system achieves 4-8% higher ARI than the no-context baseline. piRAG contributes more than MCP alone across all scenarios with zero rank inversions.

## [1:03 - 1:07] Transition to Live Demo

Now let's watch the system in action. The following shows AGRI-BRAIN processing a Heatwave scenario with real sensor data.

## [1:07 - 1:31] Live Demo (12 Phases)

Phase 1: IoT sensors extract temperature, humidity, and inventory from the spinach sensor array.
Phase 2: The PINN model computes spoilage risk using the Arrhenius equation.
Phase 3: The LSTM network forecasts demand with Bollinger band regime detection.
Phase 4: The Agent Coordinator dispatches the Farm Agent based on lifecycle stage.
Phase 5: MCP tools invoke compliance checking and SLCA lookup via JSON-RPC.
Phase 6: piRAG retrieves FDA leafy greens guidelines, physics-reranked by temperature.
Phase 7: The 5D context vector is extracted — note regulatory pressure at 1.0.
Phase 8: The policy network combines all signals into logit scores.
Phase 9: Softmax sampling selects local redistribution with 99% probability.
Phase 10: SLCA scores Carbon, Labor, Resilience, and Transparency.
Phase 11: The causal engine generates a BECAUSE/WITHOUT explanation with KB citations.
Phase 12: Evidence is hashed into a Merkle tree and anchored on blockchain.
Decision complete — all 12 phases executed with real sensor data.

## [1:31 - 1:35] Transition to Agent Theater

Now watch all 5 agents process the same Heatwave scenario, each with their own mandate and bias.

## [1:35 - 2:05] Agent Theater

The Farm Agent detects the temperature violation via MCP compliance check. piRAG retrieves FDA guidelines. It selects local redistribution with 99% probability and sends a reroute request.

The Processor Agent receives the reroute signal. It confirms the compliance violation and selects cold chain transport for processing efficiency.

The Cooperative Agent reviews equity across all SLCA pillars. It selects local redistribution to balance community resilience.

The Distributor Agent optimizes logistics under high compliance severity. It selects cold chain for controlled long-haul transport.

The Recovery Agent analyzes waste valorization options. It selects recovery to capture residual value through food banks and composting.

All 5 agents have processed the heatwave crisis, each contributing their specialized perspective to the supply chain response.

## [2:05 - 2:11] Closing

AGRI-BRAIN achieves 73.7% ARI improvement, 76.1% waste reduction, 52.5% carbon reduction, and 94.9% rerouting efficiency compared to static baselines. The full source code is available on GitHub.

---

**Total duration: ~2 minutes 15 seconds**

**TTS Settings (recommended):**
- Voice: Professional, clear male or female
- Speed: 0.9x to 1.0x (slightly slower for clarity)
- Output: MP3 or WAV
- Then combine audio + video using any video editor (CapCut, DaVinci Resolve, iMovie, or ffmpeg: `ffmpeg -i video.mp4 -i narration.mp3 -c:v copy -c:a aac output.mp4`)

// Generated English preview edition.
// Editorial navigation over the existing capability snapshot; not a runtime registry.
window.CHEMWORLD_CAPABILITIES = {
  "snapshot": "2026-10-24",
  "registry": {
    "modules": 31,
    "components": 38,
    "operations": 51,
    "measurements": 27
  },
  "tree": [
    {
      "id": "physics",
      "name": "Physical world",
      "kind": "layer",
      "preview": "",
      "children": [
        {
          "id": "foundation",
          "name": "Shared scientific foundations",
          "kind": "domain",
          "preview": "Identity, dimensions, numerics & scientific data",
          "children": [
            {
              "id": "identity-family",
              "name": "Material identity & provenance",
              "kind": "family",
              "preview": "Identify each modeled object and its provenance",
              "children": [
                {
                  "id": "identity",
                  "name": "Element, isotope & component identity",
                  "state": "done",
                  "ability": "Molecular formulae, CAS identifiers, component identity and versioned external sources.",
                  "boundary": "Data availability and the validity range of each property are checked separately.",
                  "example": "Load parameters and provenance using the actual component identity.",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "Data and Identification",
                  "children": [
                    {
                      "id": "identity-elements",
                      "name": "Elements & isotopes",
                      "kind": "method",
                      "type": "Data and Identification",
                      "state": "done",
                      "ability": "Query the element and isotope identity.",
                      "boundary": "Data availability and the validity range of each property are checked separately.",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "identity"
                    },
                    {
                      "id": "identity-formula",
                      "name": "Formulae & CAS identifiers",
                      "kind": "method",
                      "type": "Data and Identification",
                      "state": "done",
                      "ability": "Molecular formula analysis and CAS identification.",
                      "boundary": "Data availability and the validity range of each property are checked separately.",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "identity"
                    },
                    {
                      "id": "identity-sources",
                      "name": "Versioned sources",
                      "kind": "method",
                      "type": "Data and Identification",
                      "state": "done",
                      "ability": "Maintain component identifiers and versions from external sources.",
                      "boundary": "Data availability and the validity range of each property are checked separately.",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "identity"
                    }
                  ]
                }
              ]
            },
            {
              "id": "quantities-family",
              "name": "Dimensions, references & conservation",
              "kind": "family",
              "preview": "The units and historical temperature scales are considered common scientific inputs.",
              "children": [
                {
                  "id": "conservation",
                  "name": "Units, inventory & conservation",
                  "state": "done",
                  "ability": "Material, phase and energy ownership; actual transfers and explicit reference states.",
                  "boundary": "An unspecified caloric model does not automatically provide energy closure.",
                  "example": "After adding materials, transferring, and taking samples, verify the remaining materials.",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "Physical constraints"
                },
                {
                  "id": "temperature",
                  "name": "Historical temperature scales",
                  "state": "done",
                  "ability": "Five temperature scales and 500 nodes, with conversions in every direction; supported in quantities, inputs, observations and downstream capabilities.",
                  "boundary": "The differences between the published literature and the reference version are explicitly defined; the actual endpoints limit the valid domain.",
                  "example": "After converting historical temperature readings, the data is then entered into the Henry coefficient allocation or the water, light, and material properties.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Calibration method"
                }
              ]
            },
            {
              "id": "numerics-family",
              "name": "Numerical models & solvers",
              "kind": "family",
              "preview": "Collaborative advancement, algebraic constraints, and discretization.",
              "children": [
                {
                  "id": "numerics",
                  "name": "ODEs, DAEs, residuals & discretization",
                  "state": "done",
                  "ability": "Shared root finding, index-1 DAEs, derivatives, conservative discretization and events.",
                  "boundary": "The model and the domain of solution must be explicitly defined.",
                  "example": "The same fundamental solution is used for batteries, transportation, and thermodynamics.",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "Numerical methods",
                  "children": [
                    {
                      "id": "numerics-ode",
                      "name": "ODEs & events",
                      "kind": "method",
                      "type": "Numerical methods",
                      "state": "done",
                      "ability": "The model's ordinary differential equation (ODE) formulation is used to describe the system's dynamics and its response to events.",
                      "boundary": "The model and the domain of solution must be explicitly defined.",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "numerics"
                    },
                    {
                      "id": "numerics-dae",
                      "name": "Residuals & index-1 DAEs",
                      "kind": "method",
                      "type": "Numerical methods",
                      "state": "done",
                      "ability": "Shared residuals, root finding and integration of index-1 algebraic constraints.",
                      "boundary": "The model and the domain of solution must be explicitly defined.",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "numerics"
                    },
                    {
                      "id": "numerics-space",
                      "name": "Conservative discretization & derivatives",
                      "kind": "method",
                      "type": "Numerical methods",
                      "state": "done",
                      "ability": "Conserved quantities and their discrete forms, as well as model derivatives.",
                      "boundary": "The model and the domain of solution must be explicitly defined.",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "numerics"
                    }
                  ]
                }
              ]
            },
            {
              "id": "calibration-family",
              "name": "Scientific inputs & calibration",
              "kind": "family",
              "preview": "Data quality, limited fitting, and out-of-sample validation",
              "children": [
                {
                  "id": "calibration",
                  "name": "Scientific inputs & finite calibration",
                  "state": "done",
                  "ability": "Sourced scientific inputs, finite parameter fitting, temperature/parameter derivatives and identifiability diagnostics.",
                  "boundary": "Explicit parameter whitelist and fixed data grouping.",
                  "example": "Estimate thermal loss parameters using cooling curves.",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "Data and Calibration"
                }
              ]
            }
          ]
        },
        {
          "id": "properties",
          "name": "Properties & phase equilibria",
          "kind": "domain",
          "preview": "Material properties method and thermodynamic closure",
          "children": [
            {
              "id": "caloric-family",
              "name": "Caloric & phase-change properties",
              "kind": "family",
              "preview": "Heat capacity, heat generation rate, vapor pressure, and latent heat",
              "children": [
                {
                  "id": "caloric",
                  "name": "Heat capacity, formation heat & phase changes",
                  "state": "done",
                  "ability": "Methods based on heat capacity, H/S integration and inverse calculation, vapor/sublimation pressure, latent heat, and source consumption.",
                  "boundary": "Specific methods, phase comparisons, and explicit selection of temperature ranges.",
                  "example": "Calculate the heat involved in phase transitions based on temperature and composition.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Material property method"
                }
              ]
            },
            {
              "id": "bulk-properties",
              "name": "Volumetric & transport properties",
              "kind": "family",
              "preview": "Density, viscosity, and thermal conductivity are each modeled.",
              "children": [
                {
                  "id": "density",
                  "name": "Critical properties, density & virial models",
                  "state": "partial",
                  "ability": "External density/volume sources, seven B and two C virial methods, cross parameters and mixture rules; pure-gas PVT/caloric properties drive closed, fixed-volume thermal processes.",
                  "boundary": "Sources and application ranges are explicit. General mixture thermodynamics and open nonideal flow are outside the current consumer scope.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Material property method"
                },
                {
                  "id": "viscosity",
                  "name": "Molecular parameters & viscosity",
                  "state": "done",
                  "ability": "LJ: Impact integral, viscosity of pure substances and mixtures, density correction, and viscosity scale.",
                  "boundary": "The applicable domain and required parameters for each method are retained.",
                  "example": "Changes in composition affect the viscosity of gas mixtures.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Material property method"
                },
                {
                  "id": "conductivity",
                  "name": "Thermal conductivity methods",
                  "state": "done",
                  "ability": "Pure-fluid, mixture and dense-fluid correlations; water/air methods, derivatives and sourced inputs.",
                  "boundary": "Each method requires its own parameters and validity range; source identity review continues.",
                  "example": "Calculate the heat transfer response based on the actual state and the thermal conductivity coefficient.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Material property method"
                }
              ]
            },
            {
              "id": "interfacial-properties",
              "name": "Interfacial & solubility properties",
              "kind": "family",
              "preview": "Surface tension, solubility, and finite phase distribution",
              "children": [
                {
                  "id": "interfaces",
                  "name": "Surface & interfacial tension",
                  "state": "done",
                  "ability": "Pure liquid-gas, mixed, and water-oil related equations; capillary action and Laplace pressure consumption.",
                  "boundary": "Water-oil experience equation, constraints on two-phase density and composition.",
                  "example": "Changes in surface tension affect the interfacial pressure difference.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Material property method"
                },
                {
                  "id": "solubility",
                  "name": "Henry constants, solubility & solvents",
                  "state": "done",
                  "ability": "Henry: Complete calculation and mixing of derivatives, solubility of solid and liquid, Hansen parameters, and finite distribution.",
                  "boundary": "Clearly define the scope of applicability for the specified solvents, reference compounds, and neutral compounds.",
                  "example": "Gas-liquid distribution and separation of finite aqueous/organic phases.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Material property method"
                }
              ]
            },
            {
              "id": "optical-properties",
              "name": "Electrical & optical properties",
              "kind": "family",
              "preview": "Dielectric, refractive index, polarization, and concentration conversion",
              "children": [
                {
                  "id": "optics",
                  "name": "Permittivity, refraction, polarization & Brix",
                  "state": "done",
                  "ability": "Water-related equations, source curves/single point, dipole, Lorentz-Lorenz, and Brix.",
                  "boundary": "Unknown temperature measurement retained; 285 rows of automatic component mapping still need to be completed.",
                  "example": "The refractive index from the source is converted to Brix, and the dielectric constant is converted to capacitance.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Material property method"
                }
              ]
            },
            {
              "id": "fluid-equilibria",
              "name": "Fluid states & phase equilibria",
              "kind": "family",
              "preview": "EOS, phase stability, flash evaporation, and phase envelope",
              "children": [
                {
                  "id": "advanced-thermo",
                  "name": "Pure-fluid Helmholtz thermodynamics",
                  "state": "done",
                  "ability": "Native derivatives, thermodynamic properties, density roots, saturation, flash and bounded thermal consumers.",
                  "boundary": "Explicit source coefficients and validity range. Homogeneous local stability alone does not establish phase equilibrium.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Thermodynamic model"
                },
                {
                  "id": "activity-thermo",
                  "name": "Activity models & cubic mixing",
                  "state": "done",
                  "ability": "Excess and partial-molar properties, six UNIFAC variants, Twu/PPR78, GE mixing and common phase evaluation.",
                  "boundary": "Group assignments, interactions and source parameters are explicit; model agreement is not material validation.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Scientific model"
                },
                {
                  "id": "general-fluid-eos",
                  "name": "Associating & multifluid EOS",
                  "state": "partial",
                  "ability": "PC-SAFT, SAFT, association and multifluid homogeneous responses, with selected local stability, TP/caloric flash and process consumers.",
                  "boundary": "Models and parameter sources are explicit; local branch solutions do not guarantee global equilibrium for arbitrary mixtures.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Scientific model"
                },
                {
                  "id": "working-fluids",
                  "name": "Water, ice, humid air & incompressible fluids",
                  "kind": "capability",
                  "state": "finite",
                  "type": "Working-fluid properties",
                  "ability": "Independent properties and state solutions for IF97 water/steam, ice, humid air and sourced incompressible fluids.",
                  "boundary": "Explicit region, phase and parameter ranges per fluid; not coverage of every real fluid.",
                  "source": "docs/native_foundation.md"
                }
              ]
            }
          ]
        },
        {
          "id": "fuel-assessment",
          "name": "Fuels & property assessment",
          "kind": "domain",
          "preview": "Combustion calorimetry, safety, and environmental indicators",
          "children": [
            {
              "id": "combustion-family",
              "name": "Combustion thermochemistry",
              "kind": "family",
              "preview": "Product, calorific value, and flow rate specifications",
              "children": [
                {
                  "id": "combustion",
                  "name": "Combustion stoichiometry & heating values",
                  "state": "done",
                  "ability": "Complete product measurement, HHV/LHV, reference heat value, and specifications for two/three-phase flow.",
                  "boundary": "Complete-combustion balances; no flame kinetics or product-temperature solution.",
                  "example": "For a given total output rate and air-to-fuel ratio, calculate the input rate and exhaust gas composition.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Thermal chemistry calculations",
                  "children": [
                    {
                      "id": "combustion-products",
                      "name": "Product stoichiometry",
                      "kind": "method",
                      "type": "Thermal chemistry calculations",
                      "state": "done",
                      "ability": "The complete combustion product and oxygen consumption are determined by clearly defined elements.",
                      "boundary": "Complete-combustion balances; no flame kinetics or product-temperature solution.",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "combustion"
                    },
                    {
                      "id": "combustion-heat",
                      "name": "HHV / LHV",
                      "kind": "method",
                      "type": "Thermal chemistry calculations",
                      "state": "done",
                      "ability": "Calculate the high and low-end calorific values based on the specified reference conversion, and then calculate the finite heat quantity.",
                      "boundary": "Complete-combustion balances; no flame kinetics or product-temperature solution.",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "combustion"
                    },
                    {
                      "id": "combustion-streams",
                      "name": "Two-stream & three-stream specifications",
                      "kind": "method",
                      "type": "Thermal chemistry calculations",
                      "state": "done",
                      "ability": "Solve for the independent specifications of limited fuel, air, and the third flow.",
                      "boundary": "Complete-combustion balances; no flame kinetics or product-temperature solution.",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "combustion"
                    }
                  ]
                }
              ]
            },
            {
              "id": "assessment-family",
              "name": "Property & risk indicators",
              "kind": "family",
              "preview": "Evaluation of safety, environmental impact, and fuel quality.",
              "children": [
                {
                  "id": "assessment",
                  "name": "Safety & exposure assessment",
                  "state": "done",
                  "ability": "Exposure limits, classification, and rules for flammable mixtures.",
                  "boundary": "Explicitly retain source, concentration baseline, and applicable domain; not a new operating license.",
                  "example": "Compare and contrast exposure limits and flammable mixture rules under a defined concentration baseline and source.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Property assessment"
                },
                {
                  "id": "environment-assessment",
                  "name": "Environmental equivalence factors",
                  "state": "done",
                  "ability": "Calculate the equivalent amount of GWP / GTP / ODP based on emission quality and a specified time span.",
                  "boundary": "Historical sources, the time span, and prediction labels are all preserved.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Property assessment",
                  "derivedFrom": "assessment"
                },
                {
                  "id": "fuel-quality",
                  "name": "Fuel quality indicators",
                  "state": "done",
                  "ability": "A comparison of the ignition delay quality between RON and MON, based on the source ignition point.",
                  "boundary": "The sources and estimation methods are recorded separately; the actual performance of the engine is not inferred.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Property assessment",
                  "derivedFrom": "assessment"
                }
              ]
            }
          ]
        },
        {
          "id": "reaction",
          "name": "Reactions & catalysis",
          "kind": "domain",
          "preview": "Kinetic models and reaction systems",
          "children": [
            {
              "id": "reaction-models",
              "name": "Reaction networks & thermal coupling",
              "kind": "family",
              "preview": "Statement regarding the reaction network and its thermal feedback.",
              "children": [
                {
                  "id": "thermal-reaction",
                  "name": "Finite thermal reactions & gas mechanisms",
                  "state": "finite",
                  "ability": "Explicitly define the reaction network, the relationship between temperature feedback and the declared pressure-related rate.",
                  "boundary": "The mechanisms and parameters described above; not arbitrary reaction prediction.",
                  "example": "Adding the reactants, heating, waiting, and measuring the reaction progress.",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "Reaction model"
                }
              ]
            },
            {
              "id": "catalysis-models",
              "name": "Surface catalysis models",
              "kind": "family",
              "preview": "Position-based mechanisms and descriptor methods",
              "children": [
                {
                  "id": "surface",
                  "name": "Single/multisite mean-field catalysis",
                  "state": "finite",
                  "ability": "Finite sites, adsorption/reaction, site-to-site exchange, interactions, and heat transfer.",
                  "boundary": "Mechanism of the averaged field; non-spatial kMC or realistic catalyst fitting.",
                  "example": "CO/O₂: Intervention for Surface Reaction and Coverage.",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "Catalytic model"
                },
                {
                  "id": "descriptor-catalysis",
                  "name": "Descriptors, surface steady states & rate control",
                  "state": "finite",
                  "ability": "Explicit surface thermochemistry, descriptor relations, finite-condition steady maps and rate-control analysis.",
                  "boundary": "Bounded by declared sites, networks, parameters and steady branches; not an empirical catalyst calibration.",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "Catalytic model"
                }
              ]
            },
            {
              "id": "reaction-generation",
              "name": "Mechanism & reacting-flow extensions",
              "kind": "family",
              "preview": "The complete mechanism and open reaction system",
              "children": [
                {
                  "id": "kinetic-expansion",
                  "name": "Mechanism generation & finite reactor networks",
                  "state": "partial",
                  "ability": "Imported gas/surface mechanisms, molecular graphs and rules, core–edge growth, RRKM/energy-grain/QSD, finite gas networks and sensitivities.",
                  "boundary": "Uses finite rule and parameter libraries; complete Cantera/RMG workflows and databases are not covered.",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "Reaction system"
                },
                {
                  "id": "flame-catalysis",
                  "name": "1D multicomponent reacting transport",
                  "state": "finite",
                  "ability": "Finite-volume reaction, species diffusion, Stefan–Maxwell fluxes and finite feed/collector stocks.",
                  "boundary": "Restricted to the declared low-Mach transport and boundary models; general freely propagating flames are not delivered.",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "Reaction flow model"
                }
              ]
            }
          ]
        },
        {
          "id": "separation",
          "name": "Separation & particle processes",
          "kind": "domain",
          "preview": "Separation unit and limited inventory changes",
          "children": [
            {
              "id": "phase-separation",
              "name": "Equilibrium separation",
              "kind": "family",
              "preview": "Vapor-liquid extraction, liquid-liquid extraction, and distillation unit",
              "children": [
                {
                  "id": "flash",
                  "name": "Finite TP/PH flash",
                  "state": "finite",
                  "ability": "The finite phase inventory under explicit vapor-liquid equilibrium and heat transfer is closed.",
                  "boundary": "Declaration of components and model; non-generic, multi-phase, global solution.",
                  "example": "Adjust the feed heat, and observe the liquid-to-gas ratio.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Separation model"
                },
                {
                  "id": "extraction",
                  "name": "Liquid–liquid equilibrium, extraction & washing",
                  "state": "finite",
                  "ability": "Liquid-phase stability, finite two-phase distribution, mixing/settling/separation/washing.",
                  "boundary": "Supported components, activity models, and the composition domain.",
                  "example": "The discovery of the distribution pattern is linked to washing with a finite amount of solvent.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Separation model"
                },
                {
                  "id": "distillation",
                  "name": "Dynamic distillation & vacuum concentration",
                  "state": "finite",
                  "ability": "The tower plate/receiver holds the liquid, facilitates reflux, performs timed cutting, and concentrates the mixture.",
                  "boundary": "Small, explicit device; non-arbitrary industrial tower design.",
                  "example": "The effect of the return flow intervention influences the yield and recovery rate.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Separation model"
                }
              ]
            },
            {
              "id": "particle-separation",
              "name": "Crystallization & adsorption",
              "kind": "family",
              "preview": "Particle Swarm, Filtering, and Interface Inventory",
              "children": [
                {
                  "id": "crystal",
                  "name": "Crystallization, filtration & redissolution",
                  "state": "finite",
                  "ability": "Crystal seeds, limited cooling growth/dissolution, particle swarm, filtration and resuspension.",
                  "boundary": "The defined particle swarm and material model.",
                  "example": "Cooling → Growth → Filtration → Reheating and Dissolution.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Separation model"
                },
                {
                  "id": "adsorption",
                  "name": "Adsorbent contact & regeneration",
                  "state": "finite",
                  "ability": "Competitive adsorption kinetics, finite capacity, washing and inventory management.",
                  "boundary": "Clearly define the adsorption system, temperature range, and the heat of the carrier.",
                  "example": "The same adsorbent is repeatedly contacted and washed.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Separation model"
                }
              ]
            }
          ]
        },
        {
          "id": "process-systems",
          "name": "Transport & process systems",
          "kind": "domain",
          "preview": "Flow paths, heat transfer, and process organization",
          "children": [
            {
              "id": "transport-units",
              "name": "Finite flow paths & heat transfer",
              "kind": "family",
              "preview": "Pump, pipe, tank, and heat exchanger",
              "children": [
                {
                  "id": "transport",
                  "name": "Continuous flow, pumps, pipes & heat exchange",
                  "state": "finite",
                  "ability": "Finite-source tank/pump/pipe/receiver, pressure drop, shaft power, and explicit heat exchange.",
                  "boundary": "Limited work points and already declared open traffic routes.",
                  "example": "The intervention involves adjusting flow rate, residence time, and temperature simultaneously.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Transport model"
                },
                {
                  "id": "fluid-engineering",
                  "name": "Engineering fluid calculations",
                  "state": "done",
                  "ability": "Meters, valves, friction, fittings, pumps, gas/two-phase flow, packed beds/towers, open channels and equipment geometry.",
                  "boundary": "Individual correlations and finite device consumers; not an unrestricted process-network solver.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Scientific model"
                }
              ]
            },
            {
              "id": "flowsheet-models",
              "name": "General flowsheet modeling",
              "kind": "family",
              "preview": "Strict units, circuits, and optimization",
              "children": [
                {
                  "id": "flowsheet",
                  "name": "Serial units, single recycle & process design",
                  "state": "finite",
                  "ability": "Molecular thermodynamics drives compression/expansion, heat exchange and flash, serial paths, single recycle, finite receivers and bounded design.",
                  "boundary": "Explicit feed, feasible domain and local branches; not an arbitrary-topology flowsheet solver.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Process model"
                }
              ]
            }
          ]
        },
        {
          "id": "materials",
          "name": "Materials & electrochemistry",
          "kind": "domain",
          "preview": "Electrodes, batteries, and solid-state models",
          "children": [
            {
              "id": "electrode-models",
              "name": "Electrode kinetics",
              "kind": "family",
              "preview": "Response of finite electrodes and double-layer",
              "children": [
                {
                  "id": "electrode",
                  "name": "Finite electrochemistry & transient electrodes",
                  "state": "finite",
                  "ability": "Potentially driven by voltage/current, conversion, finite double-layer, and transient reading.",
                  "boundary": "The electrochemical kinetics described in the statement are applicable to finite systems.",
                  "example": "Set the potential and observe the conversion and the corresponding electrical response.",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "Electrode model"
                }
              ]
            },
            {
              "id": "battery-models",
              "name": "Battery models & diagnostics",
              "kind": "family",
              "preview": "Particles, electrolyte, spatial coupling, and impedance",
              "children": [
                {
                  "id": "spm",
                  "name": "SPM, thermal cycling & finite SEI",
                  "state": "finite",
                  "ability": "Granular diffusion, circulation, and synthesis heat model, along with retention and decay.",
                  "boundary": "The synthesis parameters are finite; this does not represent any real battery.",
                  "example": "The process involves charging and discharging, temperature increase, and maintaining multiple cycles.",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "Battery model",
                  "children": [
                    {
                      "id": "battery-spm-model",
                      "name": "SPM particle diffusion",
                      "kind": "method",
                      "type": "Battery model",
                      "state": "finite",
                      "ability": "Diffusion and voltage response of individual battery particles.",
                      "boundary": "The synthesis parameters are finite; this does not represent any real battery.",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spm"
                    },
                    {
                      "id": "battery-thermal-cycle",
                      "name": "Thermal models & cycling",
                      "kind": "method",
                      "type": "Battery model",
                      "state": "finite",
                      "ability": "Promote the cyclical process and temperature increase under the synthetic heat model described in the statement.",
                      "boundary": "The synthesis parameters are finite; this does not represent any real battery.",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spm"
                    },
                    {
                      "id": "battery-sei",
                      "name": "Finite SEI degradation",
                      "kind": "method",
                      "type": "Battery model",
                      "state": "finite",
                      "ability": "Maintain a finite SEI state, along with material and cyclical decay status.",
                      "boundary": "The synthesis parameters are finite; this does not represent any real battery.",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spm"
                    }
                  ]
                },
                {
                  "id": "spme-dfn",
                  "name": "SPMe & DFN",
                  "state": "done",
                  "ability": "Electrolyte transport and spatial coupling reactions are used to power native batteries.",
                  "boundary": "Explicit isothermal model; DFN possesses fixed synthesis parameters/grid qualification.",
                  "example": "Compare the concentration under varying current changes with the end voltage.",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "Battery model",
                  "children": [
                    {
                      "id": "battery-spme-model",
                      "name": "SPMe",
                      "kind": "method",
                      "type": "Battery model",
                      "state": "done",
                      "ability": "Explicit isothermal electrolyte transport and native propulsion.",
                      "boundary": "Explicit isothermal model; DFN possesses fixed synthesis parameters/grid qualification.",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spme-dfn"
                    },
                    {
                      "id": "battery-dfn-model",
                      "name": "DFN",
                      "kind": "method",
                      "type": "Battery model",
                      "state": "done",
                      "ability": "Coupling of space reactions with electrolyte transport; fixing synthesis parameters and grid qualification.",
                      "boundary": "Explicit isothermal model; DFN possesses fixed synthesis parameters/grid qualification.",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spme-dfn"
                    }
                  ]
                },
                {
                  "id": "impedance",
                  "name": "Impedance & extended battery models",
                  "state": "partial",
                  "ability": "Shared frequency response and read-only SPM/SPMe/DFN impedance; Thevenin RC/two-node thermal models connect to persistent devices, shared time, in-situ measurements and replay.",
                  "boundary": "SPM has independent frequency-domain qualification; other model scopes are stated individually. Thevenin uses empirical parameters and separate charge/heat balances without inferring chemical species.",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "Battery diagnostics"
                }
              ]
            },
            {
              "id": "battery-aging-family",
              "name": "Battery life & aging",
              "kind": "family",
              "preview": "Coupled aging and long-term extension",
              "children": [
                {
                  "id": "battery-aging",
                  "name": "Finite plating, SEI & deactivation",
                  "state": "partial",
                  "ability": "Finite active/metallic lithium stocks, reversible/irreversible plating and selected battery-family side-reaction coupling.",
                  "boundary": "Some independent qualification blocks retain failures; no universal lifetime prediction or complete long-term cycling validation.",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "Aging model"
                }
              ]
            },
            {
              "id": "solid-models",
              "name": "Solid-phase thermodynamics",
              "kind": "family",
              "preview": "Local phase response and global phase selection",
              "children": [
                {
                  "id": "calphad",
                  "name": "Solid solutions, TDB/CEF & MQMQA",
                  "state": "done",
                  "ability": "Multi-array, ChemSage, internal composition, derivatives, and finite thermal range.",
                  "boundary": "Single-phase localized response; three instances of conflicting original data have been explicitly rejected.",
                  "example": "Heat/cool the material and observe changes within the phase.",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "Solid-state thermodynamics model",
                  "children": [
                    {
                      "id": "calphad-cef",
                      "name": "TDB / CEF",
                      "kind": "method",
                      "type": "Solid-state thermodynamics model",
                      "state": "done",
                      "ability": "Declare the properties of the亚点阵, the internal composition, the derivative, and the local thermal response.",
                      "boundary": "Single-phase localized response; three instances of conflicting original data have been explicitly rejected.",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "calphad"
                    },
                    {
                      "id": "calphad-mqmqa",
                      "name": "ChemSage / MQMQA",
                      "kind": "method",
                      "type": "Solid-state thermodynamics model",
                      "state": "done",
                      "ability": "Clearly define the data and the local phase model within the specified parameter range.",
                      "boundary": "Single-phase localized response; three instances of conflicting original data have been explicitly rejected.",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "calphad"
                    }
                  ]
                },
                {
                  "id": "phase-diagram",
                  "name": "Multiphase CEF & binary/ternary diagrams",
                  "state": "finite",
                  "ability": "Declared multiphase CEF TP/PH equilibrium, latent heat, binary tracing and ternary isothermal nodes, tie lines and three-phase regions.",
                  "boundary": "Finite phase sets and searches retain failed nodes; no global uniqueness or infinite-resolution continuous boundary guarantee.",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "Solid-state thermodynamics model"
                }
              ]
            }
          ]
        },
        {
          "id": "water",
          "name": "Water chemistry & membrane processes",
          "kind": "domain",
          "preview": "Separation of aqueous phase, minerals, and membranes",
          "children": [
            {
              "id": "aqueous-models",
              "name": "Aqueous & mineral reactions",
              "kind": "family",
              "preview": "Equilibrium and Finite Surface Area Dynamics",
              "children": [
                {
                  "id": "aqueous",
                  "name": "Finite aqueous & mineral equilibria",
                  "state": "finite",
                  "ability": "Statement of equilibrium between acid-base/complexation and carbonation/calcite under standard conditions.",
                  "boundary": "Limited species and ion concentration range; not a complete chemical database.",
                  "example": "Add a finite amount of acid and observe the pH and mineral content.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Water-phase model"
                },
                {
                  "id": "mineral",
                  "name": "Calcite dissolution/precipitation kinetics",
                  "state": "finite",
                  "ability": "Fast water-phase equilibrium, slow mineral dissolution rate, and realistic surface area inventory.",
                  "boundary": "Fixed at 298.15 K; no spontaneous mineral nucleation observed.",
                  "example": "The reaction proceeds through changes in surface area and saturation.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Mineral kinetics"
                }
              ]
            },
            {
              "id": "membrane-models",
              "name": "Membrane separation & water treatment",
              "kind": "family",
              "preview": "RO: Pollution, Cleaning, and Process Expansion",
              "children": [
                {
                  "id": "membrane",
                  "name": "Finite RO, membrane fouling & flushing",
                  "state": "finite",
                  "ability": "Rare NaCl: reverse osmosis, salt/water permeability, deposition resistance, and limited wash solution.",
                  "boundary": "Dilute solutions at 298.15 K, forward driving force and a simplified fouling model.",
                  "example": "Repeat filtration → Contamination → Limited washing.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Membrane model"
                },
                {
                  "id": "water-expansion",
                  "name": "Sourced aqueous chemistry & finite treatment",
                  "state": "partial",
                  "ability": "Sourced aqueous/mineral models, nonideal activities, reactive transport and headspace; RO/NF/MD/ED, adsorption/resin/GAC, clarification and cake washing.",
                  "boundary": "Each model retains salinity, temperature, pressure, phase and feed limits; not complete Reaktoro/WaterTAP coverage.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Water treatment model"
                }
              ]
            }
          ]
        },
        {
          "id": "bio",
          "name": "Biochemical & stochastic networks",
          "kind": "domain",
          "preview": "Network statements, historical data, and analysis",
          "children": [
            {
              "id": "stochastic-models",
              "name": "Stochastic kinetics",
              "kind": "family",
              "preview": "Low-copy integer reaction network",
              "children": [
                {
                  "id": "ssa",
                  "name": "Integer Gillespie SSA",
                  "state": "finite",
                  "ability": "Gillespie direct, Gibson–Bruck next-reaction, explicit-partition hybrid ODE/jump dynamics and bounded tau-leap.",
                  "boundary": "Separate algorithm contracts; tau-leap is approximate, without arbitrary automatic partitioning or spatial kMC.",
                  "example": "Low-copy-number A⇌B reactions or enzyme binding and catalysis.",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "Stochastic model"
                }
              ]
            },
            {
              "id": "biochemical-format",
              "name": "Model declarations & exchange",
              "kind": "family",
              "preview": "Safety statement, mathematical expression, and round-trip",
              "children": [
                {
                  "id": "sbml-declaration",
                  "name": "Safe SBML declarations & round trips",
                  "state": "done",
                  "ability": "The secure mathematical expressions, units, and declarations have been successfully encoded and decoded.",
                  "boundary": "The readability and writeability of the model should be verified separately from its actual execution.",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "Model format"
                }
              ]
            },
            {
              "id": "biochemical-execution",
              "name": "Biochemical execution & analysis",
              "kind": "family",
              "preview": "Multi-compartment, event, and system analysis",
              "children": [
                {
                  "id": "sbml-execution",
                  "name": "Compartments, rules & event execution",
                  "state": "partial",
                  "ability": "SBML: Collaborative access for network and prioritized/delayed event management.",
                  "boundary": "Development of the branching is in progress; SED-ML/COMBINE and full qualification are pending delivery.",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "Biochemical execution"
                },
                {
                  "id": "bio-analysis",
                  "name": "Biochemical steady states, fitting & finite-time analysis",
                  "state": "finite",
                  "ability": "SBML steady/control analysis, parameter fitting and profiles, elementary flux modes and finite-time perturbations within the conservation subspace.",
                  "boundary": "Restricted to supported reaction ODEs/rules; finite-time growth is not evidence of asymptotic chaos.",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "System analysis"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "id": "runtime",
      "name": "Experiment runtime",
      "kind": "layer",
      "preview": "",
      "children": [
        {
          "id": "actions",
          "name": "Experimental operations & control",
          "kind": "domain",
          "preview": "Applying operations to materials and equipment.",
          "children": [
            {
              "id": "material-actions",
              "name": "Material operations",
              "kind": "family",
              "preview": "Container, addition of materials, routing, and sampling",
              "children": [
                {
                  "id": "handling",
                  "name": "Basic experimental actions",
                  "state": "done",
                  "ability": "Container selection/creation, adding reactants, adding solvent/catalyst, routing, transfer, sampling, and termination.",
                  "boundary": "Check the status of \"installed,\" \"task allowed,\" and \"currently executable.\"",
                  "example": "The materials are transferred from the source bottle into the reactor, and then divided into sample bottles.",
                  "source": "docs/operations.md",
                  "kind": "capability",
                  "type": "Experimental procedure"
                }
              ]
            },
            {
              "id": "device-actions",
              "name": "Device settings & control",
              "kind": "family",
              "preview": "The control contract for the equipment has been declared.",
              "children": [
                {
                  "id": "devices",
                  "name": "Finite device control",
                  "state": "finite",
                  "ability": "Heating element, flow path/pump, cooling/crystallization, electrode/power supply, and separation operation as described in the documentation.",
                  "boundary": "Simulation device contract; does not equal real hardware driver.",
                  "example": "The same heating device serves different systems sequentially.",
                  "source": "docs/operations.md",
                  "kind": "capability",
                  "type": "Device control"
                }
              ]
            }
          ]
        },
        {
          "id": "instruments",
          "name": "Instruments & measurement",
          "kind": "domain",
          "preview": "Instrument Response and Observable Channels",
          "children": [
            {
              "id": "analytical-instruments",
              "name": "Analytical instruments",
              "kind": "family",
              "preview": "Spectroscopy, chromatography, particle size analysis, and process analysis",
              "children": [
                {
                  "id": "spectra",
                  "name": "Spectroscopy, chromatography & particle size",
                  "state": "finite",
                  "ability": "HPLC, GC, UV – Vis, NMR, IR, MS, particle size interface.",
                  "boundary": "The signal/waveform identified in the declaration, compared to the inversion model; not a general spectral library identification.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Instrument response",
                  "children": [
                    {
                      "id": "spectra-optical",
                      "name": "UV–Vis / IR",
                      "kind": "method",
                      "type": "Instrument response",
                      "state": "finite",
                      "ability": "The spectral signal from the statement is compared to the inverse model.",
                      "boundary": "The signal/waveform identified in the declaration, compared to the inversion model; not a general spectral library identification.",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    },
                    {
                      "id": "spectra-structure",
                      "name": "NMR / MS",
                      "kind": "method",
                      "type": "Instrument response",
                      "state": "finite",
                      "ability": "Response from MRI and mass spectrometry, as stated.",
                      "boundary": "The signal/waveform identified in the declaration, compared to the inversion model; not a general spectral library identification.",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    },
                    {
                      "id": "spectra-chromatography",
                      "name": "HPLC / GC",
                      "kind": "method",
                      "type": "Instrument response",
                      "state": "finite",
                      "ability": "Declare the peak shape and its corresponding response in the chromatographic analysis of the known components.",
                      "boundary": "The signal/waveform identified in the declaration, compared to the inversion model; not a general spectral library identification.",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    },
                    {
                      "id": "spectra-particle",
                      "name": "Particle-size measurement",
                      "kind": "method",
                      "type": "Instrument response",
                      "state": "finite",
                      "ability": "Finite-sized response in the particle swarm state.",
                      "boundary": "The signal/waveform identified in the declaration, compared to the inversion model; not a general spectral library identification.",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    }
                  ]
                },
                {
                  "id": "programmed",
                  "name": "Programmed chromatography",
                  "state": "finite",
                  "ability": "Programs HPLC/GC, standard calibration, samples/blanks, residual and drift.",
                  "boundary": "Three synthetic standards, a finite program, and known peak templates.",
                  "example": "Standard → Sample → Blank, observe residual.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Instrument response"
                }
              ]
            },
            {
              "id": "state-instruments",
              "name": "State measurement",
              "kind": "family",
              "preview": "Physical states, electrochemistry, and discrete counting",
              "children": [
                {
                  "id": "physical-readouts",
                  "name": "Physical & phase-state readouts",
                  "state": "finite",
                  "ability": "pH, pressure/gas composition, phase state, flow rate, water phase pH, membrane, solid phase, and CALPHAD.",
                  "boundary": "The availability is determined by the corresponding state of the world and the availability of materials.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "measurements"
                },
                {
                  "id": "electrical-readouts",
                  "name": "Electrical & battery readouts",
                  "state": "finite",
                  "ability": "Electrode ammeter, battery meter.",
                  "boundary": "The declared electrode/battery models and measurement channels.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "measurements"
                },
                {
                  "id": "discrete-readouts",
                  "name": "Molecule counts & trapping",
                  "state": "finite",
                  "ability": "Discrete molecule counting, finite capture, and delay analysis.",
                  "boundary": "Low copy number/sample contract; not a substitute for continuous liquid sampling.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "measurements"
                }
              ]
            }
          ]
        },
        {
          "id": "lifecycle",
          "name": "Scheduling & state lifecycle",
          "kind": "domain",
          "preview": "Experimental time, equipment, and recalculation",
          "children": [
            {
              "id": "scheduling-family",
              "name": "Time & device scheduling",
              "kind": "family",
              "preview": "Shared resource utilization and equipment access",
              "children": [
                {
                  "id": "scheduler",
                  "name": "Shared-lab time & device queues",
                  "state": "done",
                  "ability": "Statement: System-wide synchronization of device identities, usage, and occupancy. FIFO In contrast to delayed results.",
                  "boundary": "Explicitly finite system; if the combination fails, revert to the common submission boundary.",
                  "example": "Two samples are competing for use of the same offline analysis instrument.",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "Scheduling mechanism"
                }
              ]
            },
            {
              "id": "persistent-state",
              "name": "Persistent state & recovery",
              "kind": "family",
              "preview": "Material replacement, maintenance, and fault recovery",
              "children": [
                {
                  "id": "persistence",
                  "name": "Persistent devices & partial material replacement",
                  "state": "done",
                  "ability": "Calibration, contamination, loading, aging, and actual maintenance; localized replacement of components/equipment.",
                  "boundary": "Each state has a unique owner, and historical samples are not modified by subsequent changes.",
                  "example": "Replace the materials, while retaining any contamination or residue from the previous materials or equipment.",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "Life cycle mechanisms"
                },
                {
                  "id": "continuation",
                  "name": "Snapshots & state continuation",
                  "state": "done",
                  "ability": "Physical state, transactions, snapshots and recalculations of RNG and the queue.",
                  "boundary": "Explicitly check the compatibility range of the semantics between snapshots and execution.",
                  "example": "Save at intermediate points, and resume the event and equipment status upon restoration.",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "Calculation mechanism"
                }
              ]
            }
          ]
        },
        {
          "id": "records",
          "name": "Observations & resource records",
          "kind": "domain",
          "preview": "Experimental results and actual consumption",
          "children": [
            {
              "id": "observations-family",
              "name": "Native observations & final assays",
              "kind": "family",
              "preview": "Units, quality, collection time, and final inspection",
              "children": [
                {
                  "id": "assay",
                  "name": "Final assays & quality-tagged observations",
                  "state": "done",
                  "ability": "Liquid/Gas Final Inspection, Native Units, Collection Time, Uncertainty, Missing Values, and Quality Marks.",
                  "boundary": "Unobserved and invalid values are retained; reading from the cache does not equal resampling.",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "Observation contract"
                }
              ]
            },
            {
              "id": "resource-ledger",
              "name": "Resource accounting",
              "kind": "family",
              "preview": "Time, cost, samples, and energy",
              "children": [
                {
                  "id": "resource-accounting",
                  "name": "Physical & resource ledgers",
                  "state": "done",
                  "ability": "Calculate physical time, costs, sample usage, and energy consumption.",
                  "boundary": "The resource events align with the actual execution transactions, and do not rely on cached reads as a substitute for new data acquisition.",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "Resource record",
                  "derivedFrom": "continuation"
                }
              ]
            }
          ]
        },
        {
          "id": "access",
          "name": "Interfaces & workbenches",
          "kind": "domain",
          "preview": "Program access, observation interface, and external access",
          "children": [
            {
              "id": "programmatic-access",
              "name": "Programmatic interfaces",
              "kind": "family",
              "preview": "Different calling methods share the same environment.",
              "children": [
                {
                  "id": "entrypoints",
                  "name": "Runtime／Gym／CLI／Agent",
                  "state": "done",
                  "ability": "Same experimental state, action contract, native observation, and resource record.",
                  "boundary": "The agent's weights, the training algorithm, and its private memory are located outside the environment.",
                  "source": "docs/agents.md",
                  "kind": "capability",
                  "type": "Program interface",
                  "children": [
                    {
                      "id": "interface-runtime",
                      "name": "Runtime",
                      "kind": "method",
                      "type": "Program interface",
                      "state": "done",
                      "ability": "Directly call upon the same experimental session and native observations.",
                      "boundary": "The agent's weights, the training algorithm, and its private memory are located outside the environment.",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    },
                    {
                      "id": "interface-gym",
                      "name": "Gym",
                      "kind": "method",
                      "type": "Program interface",
                      "state": "done",
                      "ability": "Use the same experimental state and action contract by resetting/stepping.",
                      "boundary": "The agent's weights, the training algorithm, and its private memory are located outside the environment.",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    },
                    {
                      "id": "interface-cli",
                      "name": "CLI",
                      "kind": "method",
                      "type": "Program interface",
                      "state": "done",
                      "ability": "Execute actions, save, and check trajectories through the command line.",
                      "boundary": "The agent's weights, the training algorithm, and its private memory are located outside the environment.",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    },
                    {
                      "id": "interface-agent",
                      "name": "Agent",
                      "kind": "method",
                      "type": "Program interface",
                      "state": "done",
                      "ability": "Use the Agent execution interface to access and record the same environment and resources.",
                      "boundary": "The agent's weights, the training algorithm, and its private memory are located outside the environment.",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    }
                  ]
                }
              ]
            },
            {
              "id": "interactive-tools",
              "name": "Interactive tools & observatories",
              "kind": "family",
              "preview": "Local experiments and read-only trajectory browsing",
              "children": [
                {
                  "id": "observatory",
                  "name": "Student Lab & Agent Observatory",
                  "state": "done",
                  "ability": "Local interactive experiment; read-only trajectory browsing, resource checking, and replay.",
                  "boundary": "The interface's operation is consistent with the supported domain of the model and the tasks.",
                  "source": "docs/student-lab.md",
                  "kind": "capability",
                  "type": "Work bench"
                }
              ]
            },
            {
              "id": "external-integration",
              "name": "External data & hardware integration",
              "kind": "family",
              "preview": "Transferring independent models to finite physical systems",
              "children": [
                {
                  "id": "bridge",
                  "name": "Real-world data, models & hardware bridges",
                  "state": "planned",
                  "ability": "Transfer and validation of independent models, real-world data, and narrow physical systems.",
                  "boundary": "Overall bridging route; do not treat existing interfaces as real laboratory connections.",
                  "source": "docs/student-lab.md",
                  "kind": "capability",
                  "type": "External access"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "id": "evaluation",
      "name": "Tasks & evaluation",
      "kind": "layer",
      "preview": "",
      "children": [
        {
          "id": "research-tasks",
          "name": "Research tasks",
          "kind": "domain",
          "preview": "Scientific questions and intervention goals",
          "children": [
            {
              "id": "discovery-tasks",
              "name": "Law discovery & characterization",
              "kind": "family",
              "preview": "Distribution laws and non-ideal equilibrium",
              "children": [
                {
                  "id": "partition-task",
                  "name": "Partition-law discovery",
                  "state": "finite",
                  "ability": "Comparison of interventions based on phase conditions, open measurement, and hidden world responses.",
                  "boundary": "Fixed-task contract and finite-phase model.",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "Research tasks"
                },
                {
                  "id": "equilibrium-task",
                  "name": "Nonideal equilibrium characterization",
                  "state": "finite",
                  "ability": "Concentration probes, pH/ UV –Vis feedback and balance principles.",
                  "boundary": "The non-ideal model and components specified in the task statement.",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "Research tasks"
                }
              ]
            },
            {
              "id": "coupled-process-tasks",
              "name": "Reaction & separation tasks",
              "kind": "family",
              "preview": "Crystallization and distillation each have their own distinct objectives.",
              "children": [
                {
                  "id": "coupled-tasks",
                  "name": "Reaction → crystallization",
                  "state": "finite",
                  "ability": "Reaction kinetics is propagated to yield, purity, and particle size.",
                  "boundary": "Scope of independent crystallization research tasks and associated models.",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "Research tasks"
                },
                {
                  "id": "distillation-task",
                  "name": "Reaction → distillation",
                  "state": "finite",
                  "ability": "Reaction kinetics is propagated to yield, purity, and fraction.",
                  "boundary": "Scope of independent distillation research tasks and models.",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "Research tasks",
                  "derivedFrom": "coupled-tasks"
                }
              ]
            },
            {
              "id": "control-tasks",
              "name": "Process control & optimization",
              "kind": "family",
              "preview": "Flow reactions and electrochemical interventions",
              "children": [
                {
                  "id": "flow-task",
                  "name": "Flow-reaction optimization",
                  "state": "finite",
                  "ability": "Flow rate, residence time, and temperature, along with the hidden rate law.",
                  "boundary": "A clearly defined, finite-flow reaction world.",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "Research tasks"
                },
                {
                  "id": "electro-task",
                  "name": "Electrochemical conversion",
                  "state": "finite",
                  "ability": "Electrochemical principles, diagnostic feedback, and intervention based on set values.",
                  "boundary": "The task specifies the allowed electrodes and measurement interfaces.",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "Research tasks"
                }
              ]
            }
          ]
        },
        {
          "id": "protocol",
          "name": "Evaluation & reproducibility",
          "kind": "domain",
          "preview": "Research on cycles, replay, and scoring.",
          "children": [
            {
              "id": "research-protocol",
              "name": "Research workflow",
              "kind": "family",
              "preview": "Design, prediction, and validation",
              "children": [
                {
                  "id": "research-loop",
                  "name": "Design, prediction & blind-testing loop",
                  "state": "finite",
                  "ability": "Experimental design, research notes, pre-execution predictions, and a record of actual results.",
                  "boundary": "Existing teaching examples have been developed; however, this does not confirm that the Agent's scientific capabilities have been proven.",
                  "source": "docs/agents.md",
                  "kind": "capability",
                  "type": "Evaluation agreement"
                }
              ]
            },
            {
              "id": "replay-protocol",
              "name": "Trajectory replay & scoring",
              "kind": "family",
              "preview": "Replay of previously recorded execution semantics",
              "children": [
                {
                  "id": "replay",
                  "name": "Replay & rescoring",
                  "state": "done",
                  "ability": "Record the actions and results, replay the trajectory, and recalculate the metrics.",
                  "boundary": "Meets the original semantic requirements; if a final inspection is missing, the final score cannot be generated.",
                  "source": "docs/agents.md",
                  "kind": "capability",
                  "type": "Replay function"
                }
              ]
            }
          ]
        },
        {
          "id": "learning-resources",
          "name": "Teaching & task resources",
          "kind": "domain",
          "preview": "Examples of research that can be reviewed",
          "children": [
            {
              "id": "notebook-resources",
              "name": "Task notebooks",
              "kind": "family",
              "preview": "Pairing, intervention, and feedback",
              "children": [
                {
                  "id": "notebooks",
                  "name": "Six task notebook families",
                  "state": "done",
                  "ability": "Candidate interventions, three sets of feedback, complete trajectories, and paired hidden worlds.",
                  "boundary": "The results of the teaching pairing and the formal benchmark are presented separately.",
                  "source": "docs/student-lab.md",
                  "kind": "capability",
                  "type": "Teaching resources"
                }
              ]
            }
          ]
        }
      ]
    }
  ],
  "delivery": {
    "id": "release",
    "name": "Final steps for cross-platform deployment and release.",
    "state": "partial",
    "ability": "Installation/example acceptance is complete; final platform compatibility testing remains to be done.",
    "boundary": "Linux/macOS qualifications need to be updated; development skills and official release are separate.",
    "source": "docs/current-foundation.md"
  },
  "taxonomy": {
    "version": 2,
    "levels": [
      "Project",
      "Architecture layer",
      "Capability domain",
      "Function group",
      "Capability",
      "Method / variant"
    ],
    "counting": "Only specific capabilities are counted; variations in methods and deliverables are not counted twice."
  }
};

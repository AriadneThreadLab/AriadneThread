# Global Horizontal Irradiance in solar assessment

This note explains what global horizontal irradiance means in photovoltaic
assessment. It is documentation for grounding language-model answers. It is not
a SolarDSS calculation, not a site indicator, and not a substitute for
deterministic resource data obtained by the assessment engine.

## Definition

Global horizontal irradiance, commonly abbreviated GHI, is the total solar
radiation received on a horizontal surface. It includes the beam component that
arrives in a straight line from the sun and the diffuse component that arrives
after scattering in the atmosphere. Related terms include direct normal
irradiance for the beam on a surface perpendicular to the sun, and diffuse
horizontal irradiance for the scattered portion on the horizontal plane. These
are names of physical quantities. Their numerical values for a given site must
come from a resource provider or a deterministic engine, never from this
corpus.

## How GHI is used as a concept

Siting methodologies use GHI as a descriptor of the solar resource available to
a horizontal collector. Yield models then apply orientation, soiling, and
module behaviour. The conceptual role of GHI is to distinguish resource quality
from land, grid, and permitting constraints. A passage that defines GHI does
not decide whether a site is acceptable. Decision-support software applies
configured thresholds and provider layers separately.

## What this document must not replace

Do not treat any sentence here as an annual irradiation figure, a typical city
resource, or a computed energy yield. If a user asks what GHI means, answer
from this definition and cite the source. If a user asks how much GHI a site
receives, that answer belongs to deterministic analytical evidence from
SolarDSS, not to retrieval-augmented documentation.

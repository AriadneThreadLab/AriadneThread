# How photovoltaic site exclusion is explained

When a decision-support run excludes a candidate, the exclusion is a
deterministic screening outcome. This note explains the categories of reasons
that methodologies typically record. It does not decide that any named parcel
is excluded and it does not contain computed indicator values.

## Common exclusion categories

Siting frameworks group exclusions into land-status, physical, and
infrastructure classes. Land-status reasons include protected areas, existing
urban fabric, and conflicting land use. Physical reasons include terrain that
the configured screen treats as unsuitable, flood or geohazard overlays, and
resource layers that fail a configured test. Infrastructure reasons include
excessive distance to roads or to the electrical network when the project
requires a grid connection. The names of these categories are documentation.
The boolean outcome for a site is analytical.

## How to answer “why was this site excluded?”

First report the exclusion code or reason string returned by SolarDSS, if one
exists. That string is deterministic analytical evidence. Then use this note
only to explain what that category means in methodological language. If the
engine did not run, say that retrieval cannot determine whether the site was
excluded. Do not guess GHI, slope, or setback distances in order to justify an
exclusion.

## Provenance rule

An explanation of policy language is knowledge evidence. An exclusion flag,
score, or indicator for a concrete candidate is analytical evidence. Mixing
the two in a single undifferentiated citation list would hide whether the
answer came from a document or from the engine.

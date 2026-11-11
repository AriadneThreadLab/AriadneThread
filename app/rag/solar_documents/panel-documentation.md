# Photovoltaic module documentation and nameplate evidence

Module datasheets and certificates describe a product, not a site. This note
explains how panel documentation is used in a decision file. It does not list
nameplate watts, efficiency percentages, or temperature coefficients, because
those values belong to the chosen product file and to the yield engine.

## What panel documentation contains

A complete module package usually includes a datasheet, installation
instructions, a declaration of conformity, and type-test certificates against
design-qualification and safety standards. The documents identify the model,
the applicable standard series, and the conditions under which ratings were
measured. Readers should copy ratings from the manufacturer file into the
engineering model. They must not invent ratings from a retrieval passage.

## How documentation relates to siting

Siting engines need a module class only when the capability actually sizes an
array. Until the user or a catalogue supplies a model, SolarDSS must not be
fed fabricated electrical parameters. This note may be cited when the user
asks what panel documentation is for. It must not be cited as the source of a
site GHI, a slope, or an annual energy figure.

## Provenance

When a datasheet is on file, quote its model identifier as product evidence.
When only this grounding note was retrieved, label the citation as knowledge
and document evidence, never as a computed indicator.

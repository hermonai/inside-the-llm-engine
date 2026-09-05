# Animation standard

Canonical JSON fixtures and explicit integer frame lists generate the static
plate and all animation frames. v1 uses HTML/SVG and a small shared player;
no proprietary tool, raster video, remote dependency, or WebGL is necessary.
RoPE changes angle; cache changes valid length; batching changes active column.
Static SVG shows all four states simultaneously. The animation highlights one
state in that complete plate; it does not invent intermediate simulation data.

Provide Play/Pause, Previous, Next, and a labeled range control. Start paused,
stop at the last frame and on hidden tabs, and advance manually under reduced
motion. Announce frame state. Every page links its static equivalent. Keyboard
and narrow-view checks accompany semantic fixture checks. Duration is a teaching
cadence, not a timing measurement. See [build commands](FIGURE_BUILD.md).

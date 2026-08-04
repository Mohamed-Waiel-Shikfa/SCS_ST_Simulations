"""What every parameter means, what it does, and what it costs.

Several of these names are not self-explanatory even to someone who knows the
physics - J against B is the classic one - and a design tool that shows a
number without saying what moving it would do is only half a tool.  Each entry
has three parts:

    what    the physical quantity
    effect  which way the design moves when this goes up
    cost    what you give up for it

These are served to the UI and shown on hover, so the explanation lives next
to the number rather than in a document nobody opens.
"""

PARAM_INFO = {
    # ---- design variables --------------------------------------------------
    "material": dict(
        what="The permanent-magnet grade. Everything commercially available "
             "with an intrinsic coercivity below 2000 kA/m is in the search.",
        effect="Sets remanence (how much force), coercivity (how well it "
               "resists a neighbour trying to reverse it), recoil "
               "permeability and density all at once.",
        cost="Coercivity buys repulsion and demagnetisation margin, and costs "
             "switching energy roughly as Hcj squared. The rare-earth grades "
             "are in the table to show where the wall is, not because they "
             "are expected to win."),
    "circuit": dict(
        what="Whether each magnet sits in a soft-iron pot core or is a bare "
             "rod.",
        effect="A pot core closes the magnetic circuit, which raises the "
               "operating point, roughly doubles the holding force, and makes "
               "the coil far more effective.",
        cost="Steel is dense, and the keeper wall eats radius that the magnet "
             "could have used."),
    "n_gon": dict(
        what="The polygon of each of the three orthogonal rings. The module "
             "has 3n - 6 faces, of which exactly 6 - the axis faces - may "
             "latch.",
        effect="The pivot angle is 360/n, so a larger n lifts the centre of "
               "mass far less when rolling: 41 % of the half-width for a "
               "cube, 2 % for a 16-gon.",
        cost="More faces means more magnets, more coils and more driver "
             "channels, so mass and switching energy both rise."),
    "r_face": dict(
        what="Distance from the module centre to a pole face. This is the "
             "module's half-size; the bounding cube is twice it.",
        effect="Sets everything else: face width, available EPM diameter, "
               "internal volume for electronics.",
        cost="Mass goes as the cube of it, and it must stay inside the 5 cm "
             "envelope."),
    "d_mag": dict(
        what="Magnet diameter.",
        effect="Pole area goes as the square, and so does force at short "
               "range. Also raises the length-to-diameter ratio's "
               "denominator, which lowers the self-demagnetising factor.",
        cost="Mass, and it competes with the winding and the keeper for the "
             "radius available on a face."),
    "l_mag": dict(
        what="Magnet length along the face normal.",
        effect="A longer rod is harder to demagnetise - the load line "
               "improves - so it holds a higher operating point.",
        cost="Depth into the module, mass, and it raises the ampere-turns "
             "needed to switch, which scale with length."),
    "t_steel": dict(
        what="Thickness of the soft-iron keeper wall around each magnet.",
        effect="Carries the return flux. Thicker steel saturates later, which "
               "is what stops a bigger capacitor from buying more field.",
        cost="Dense, and it takes radius from the magnet."),
    "r_clear": dict(
        what="Radial clearance between the winding and the steel.",
        effect="Manufacturing margin, and it controls the leakage path around "
               "the magnet.",
        cost="Pure loss of radius: it holds neither copper nor iron nor "
             "magnet."),
    "gap": dict(
        what="Working air gap between two mated pole faces.",
        effect="Force falls very steeply with it - roughly a factor of two "
               "between 0.05 and 0.3 mm.",
        cost="Making it small is a tolerance and surface-finish problem, not "
             "a design one. Any shell material in front of the pole adds to "
             "it twice over."),
    "wire_d": dict(
        what="Bare copper diameter of the winding wire.",
        effect="Thicker wire has lower resistance per turn, so more current "
               "for the same bank voltage.",
        cost="Fewer turns fit in the same space, and ampere-turns is what "
             "matters, so there is a genuine optimum rather than a "
             "direction."),
    "n_layers": dict(
        what="Number of winding layers. A real integer, not a smooth "
             "quantity.",
        effect="More layers is more turns and more ampere-turns per amp.",
        cost="Each new layer sits at a larger radius, so its turns cost more "
             "copper and are worth less: turns per ohm falls steadily with "
             "depth. The winding build also eats the radius the keeper "
             "wanted."),
    "v_cap": dict(
        what="Capacitor bank voltage.",
        effect="Peak current, and therefore peak field, is nearly linear in "
               "it until the iron saturates.",
        cost="Energy goes as the square. It also sets which MOSFETs and "
             "capacitors are available at all - above about 250 V the "
             "options thin out sharply."),
    "c_cap": dict(
        what="Capacitor bank capacitance.",
        effect="A larger bank holds the current up for longer, which matters "
               "when the coil inductance is high.",
        cost="Energy, mass and volume all go up linearly, and the bank is "
             "often the single heaviest electronic component."),
    "pulse_mode": dict(
        what="Whether the coil is driven by one capacitor discharge or by a "
             "train of shorter pulses.",
        effect="A train at the right frequency and duty can reach the same "
               "peak field for substantially less energy out of the bank, "
               "because the bank is not dumped into resistance on a single "
               "swing.",
        cost="Needs real gate timing rather than a single trigger, and the "
             "switches see more transitions."),
    "f_pulse": dict(
        what="Pulse-train frequency.",
        effect="Sets how much the coil current decays between pulses. Too "
               "slow and each pulse starts from zero; too fast and the duty "
               "cycle cannot deliver enough charge per pulse.",
        cost="Switching loss in the transistors rises with it."),
    "duty": dict(
        what="Fraction of each pulse period the bridge is conducting.",
        effect="Directly sets how much charge is delivered per cycle.",
        cost="High duty approaches the single-shot case and loses the energy "
             "advantage; low duty may never reach the threshold."),
    "n_pulses": dict(
        what="How many pulses are in the train.",
        effect="More pulses build the field further.",
        cost="Total energy and the time the manoeuvre takes."),

    # ---- fields and results -----------------------------------------------
    "J_attract": dict(
        what="Volume-averaged intrinsic POLARISATION J of the magnet in the "
             "attracting state, in tesla. J = B - mu0 H: it is the part of "
             "the flux density the material itself contributes.",
        effect="This is the quantity that actually produces force. It is not "
               "a material constant - it depends on the circuit the magnet "
               "sits in, and collapses when a neighbour pushes back.",
        cost="Do not confuse it with B. B includes the applied field, so B "
             "can be large while J is nearly zero, which is exactly what "
             "happens to a magnet being switched."),
    "J_repel": dict(
        what="The same polarisation in the repelling state, where the two "
             "magnets are demagnetising each other.",
        effect="The gap between J_attract and J_repel IS the attract/repel "
               "asymmetry. A rigid magnet would show none.",
        cost="A grade with a soft knee loses most of its J here, which is why "
             "Alnico 5 repels so poorly."),
    "n_eff": dict(
        what="Effective demagnetising factor of the magnetic circuit, "
             "measured from the field solve rather than assumed.",
        effect="Fraction of the coil's ampere-turns that is lost to the "
               "external path: the magnet only sees (1 - n_eff) of the drive. "
               "For a bare rod it is mu_rec times the shape demagnetising "
               "factor; steel drops it sharply.",
        cost="It is the single number through which the steel, the working "
             "gap, a latched neighbour and the magnet's own permeability all "
             "reach the driver."),
    "margin": dict(
        what="Worst |H| / Hcj the magnet sees over both operating states.",
        effect="How close the magnet is to erasing itself in normal service.",
        cost="Above about 0.8 the loss is irreversible and cumulative, so it "
             "is a hard constraint rather than something to trade."),
    "F_attract": dict(
        what="Axial force between two mated modules with opposite poles "
             "facing.",
        effect="Sets the holding force, which must carry several times the "
               "module's own weight.",
        cost="Comes from the same magnet that has to be switched, so it "
             "trades directly against switching energy."),
    "F_repel": dict(
        what="Axial force with like poles facing.",
        effect="This is what drives locomotion. Without it a module can latch "
               "but not move.",
        cost="Always much smaller than attraction, because in this state the "
             "magnets partly demagnetise each other."),
    "asymmetry": dict(
        what="F_attract divided by F_repel.",
        effect="How lopsided the magnet is. A perfectly rigid magnet pair "
               "gives 1.014; real grades give 3 to 10.",
        cost="Minimising it is an objective, because a design that latches "
             "hard and cannot push is useless for locomotion."),
    "h_peak": dict(
        what="Peak field the coil drives into the magnet during the pulse.",
        effect="Must exceed about three times the intrinsic coercivity to "
               "reverse the magnet reliably.",
        cost="Everything: bank voltage squared, coil copper, and driver "
             "components rated for the peak current."),
    "b_steel_peak": dict(
        what="Peak flux density in the keeper annulus during switching.",
        effect="Below about 1.9 T the iron is nearly free and helps. Above "
               "it, the iron stops carrying incremental flux.",
        cost="Once saturated, extra ampere-turns buy very little extra field, "
             "so a bigger capacitor stops helping."),
    "pivot_ratio": dict(
        what="Magnetic work available over one roll, divided by the "
             "gravitational barrier of lifting the centre of mass to the "
             "vertex radius.",
        effect="Must exceed 1 for the roll to be possible at all, and is "
               "required to exceed 1.5 for margin.",
        cost="Necessary but not sufficient - it is an energy bound, and "
             "MuJoCo is the arbiter of whether the manoeuvre actually "
             "works."),
    "hold_ratio": dict(
        what="Attraction divided by the module's own weight.",
        effect="How many modules can hang off one joint.",
        cost="Required to exceed 3."),
    "e_switch": dict(
        what="Bank energy needed to switch every face of the module once.",
        effect="Sets the battery size and how many moves a charge buys.",
        cost="Scales with coercivity squared and with face count, so it is "
             "the main thing pushing back on high-n designs and hard "
             "grades."),
    "scalar": dict(
        what="A single combined score, used only for ranking in lists.",
        effect="Geometric blend of the five objectives, zero for any "
               "infeasible design.",
        cost="It hides the trade-off. The Pareto front is the real answer; "
             "this is a convenience."),
    "free_volume": dict(
        what="Internal volume left for electronics after the shell and all "
             "the EPM assemblies are removed, with a packing efficiency "
             "applied.",
        effect="Must exceed what the driver actually needs.",
        cost="Falls fast as the EPMs get bigger or more numerous."),
    "m_module": dict(
        what="Total module mass: magnets, coils, steel, shell, capacitors, "
             "battery and board.",
        effect="Sets the gravitational barrier for every manoeuvre and the "
               "holding force required.",
        cost="An objective to minimise; it fights against almost everything "
             "else."),
}

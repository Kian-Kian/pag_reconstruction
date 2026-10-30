# Input format

CTF inputs must describe a rectangular grid with `XCells`, `YCells`, `XStep` and `YStep`. Numeric rows must appear in row-major grid order, with X varying fastest. The reader does not reorder rows. X and Y are in micrometres; Euler angles are in degrees. Use a square grid, as in the original study. Other grid sizes and step sizes can be read, but the released models were trained on 400 by 400 maps with a 0.15 micrometre step and their accuracy at other scales has not been established.

Required tab-separated columns are `Phase`, `X`, `Y`, `Bands`, `Error`, `Euler1`, `Euler2`, `Euler3`, `MAD`, `BC` and `BS`. Extra columns are retained during CTF export. Quality fields are used as model features and should follow the CTF acquisition conventions of the original workflow; missing values cannot be replaced arbitrarily without changing the model input distribution. All numeric values must be finite.

Bunge Euler angles produce specimen-to-crystal rotation matrices. Cubic proper rotations act on the left. The parent-to-child orientation relationship satisfies `R_alpha = OR @ R_gamma`.

Inference selects only the requested alpha phase ID. Unindexed pixels, other phases and connected components smaller than two pixels remain excluded. Connectivity uses horizontal and vertical neighbors. No filling, smoothing or reassignment is performed.

Training pairs must have equal shapes, registered pixel coordinates and matching measurement fields. The paired reader expects alpha phase ID 1 and reference gamma phase ID 2. Gamma maps supply supervision and OR calibration only during preparation and training. High-confidence labels require parent area of at least 50 pixels, overlap coverage of at least 0.90 and parent-assignment purity of at least 0.95.

`reconstruction.npz` uses zero-based grain and PAG identifiers; excluded pixels have label -1. `component_ids` gives the ordered IDs associated with `component_orientation`. The latter contains 3 by 3 specimen-to-parent-crystal matrices. Node-level orientation diagnostics remain in alpha-grain order.

extends Node3D

# =============================================================================
# ReplayLoader.gd — Warehouse simulation replay visualiser
# =============================================================================
# Camera (mouse only — no keyboard focus needed):
#   Left drag       — orbit (rotate around warehouse)
#   Left drag + PAN — pan   (click "ORBIT mode" button to switch)
#   Scroll wheel    — zoom in / out
#   Middle drag     — pan (always)
# =============================================================================

@export var replay_path    : String = "res://replay.json"
@export var playback_speed : float  = 10.0
@export var cell_size_m    : float  = 1.0

const COLOR_WAITING  = Color(0.50, 0.55, 0.65)
const COLOR_TO_ITEM  = Color(0.20, 0.88, 0.38)
const COLOR_PICKING  = Color(1.00, 0.78, 0.08)
const COLOR_TO_DEPOT = Color(0.20, 0.58, 1.00)
const COLOR_RESTING  = Color(0.75, 0.28, 0.95)
const COLOR_DONE     = Color(0.70, 0.72, 0.75)

var _frames       : Array      = []
var _meta         : Dictionary = {}
var _layout       : Dictionary = {}
var _current_tick : float      = 0.0
var _agent_nodes  : Array      = []
var _agent_mats   : Array      = []
var _paused       : bool       = false

# ── Camera spherical state ────────────────────────────────────────────────────
var _pivot        : Node3D   = null
var _camera       : Camera3D = null
var _cam_dist     : float    = 30.0
var _cam_azimuth  : float    = 180.0
var _cam_elev     : float    = 55.0

# ── Mouse state ───────────────────────────────────────────────────────────────
var _lmb_held     : bool     = false
var _mmb_held     : bool     = false
var _pan_mode     : bool     = false

# ── Scene-defined UI nodes (set up in Warehouse.tscn, connected in _ready) ────
@onready var _label_tick  : Label  = $HUD/LabelTick
@onready var _label_info  : Label  = $HUD/LabelInfo
@onready var _btn_pause   : Button = $HUD/BtnPause
@onready var _btn_restart : Button = $HUD/BtnRestart
@onready var _btn_faster  : Button = $HUD/BtnFaster
@onready var _btn_slower  : Button = $HUD/BtnSlower
@onready var _btn_mode    : Button = $HUD/BtnMode


# =============================================================================
# Lifecycle
# =============================================================================

func _ready():
	_connect_buttons()
	_load_replay()
	if not _frames.is_empty():
		_build_scene()
		_update_agents(0)
		_update_hud(0)


func _process(delta):
	if _frames.is_empty() or _paused:
		return
	_current_tick += playback_speed * delta
	var tick_idx = int(_current_tick)
	if tick_idx >= _frames.size():
		_current_tick = 0.0
		tick_idx = 0
	_update_agents(tick_idx)
	_update_hud(tick_idx)


func _unhandled_input(event):
	# _unhandled_input fires only after UI buttons have had first chance —
	# so button clicks are never accidentally treated as camera drags.
	if event is InputEventMouseButton:
		match event.button_index:
			MOUSE_BUTTON_LEFT:
				_lmb_held = event.pressed
			MOUSE_BUTTON_MIDDLE:
				_mmb_held = event.pressed
			MOUSE_BUTTON_WHEEL_UP:
				_cam_dist = max(_cam_dist * 0.88, 4.0)
				_apply_camera()
			MOUSE_BUTTON_WHEEL_DOWN:
				_cam_dist = min(_cam_dist * 1.14, 200.0)
				_apply_camera()

	if event is InputEventMouseMotion:
		if _lmb_held:
			if _pan_mode:
				_pan_mouse(event.relative)
			else:
				_cam_azimuth -= event.relative.x * 0.35
				_cam_elev     = clamp(_cam_elev - event.relative.y * 0.25, 10.0, 88.0)
				_apply_camera()
		elif _mmb_held:
			_pan_mouse(event.relative)


# =============================================================================
# Button connections  (buttons defined in Warehouse.tscn)
# =============================================================================

func _connect_buttons():
	print("=== Button check ===")
	print("  BtnPause   : ", _btn_pause)
	print("  BtnRestart : ", _btn_restart)
	print("  BtnFaster  : ", _btn_faster)
	print("  BtnSlower  : ", _btn_slower)
	print("  BtnMode    : ", _btn_mode)
	if _btn_pause == null:
		push_error("Buttons not found — check scene node paths in Warehouse.tscn")
		return
	_btn_pause.pressed.connect(func():
		print(">> PAUSE clicked")
		_paused = !_paused
		_btn_pause.text = "▶  Resume" if _paused else "⏸  Pause"
	)

	_btn_restart.pressed.connect(func():
		print(">> RESTART clicked")
		_current_tick = 0.0
		_paused       = false
		_btn_pause.text = "⏸  Pause"
	)

	_btn_faster.pressed.connect(func():
		print(">> FASTER clicked")
		playback_speed = min(playback_speed * 1.5, 720.0)
	)

	_btn_slower.pressed.connect(func():
		print(">> SLOWER clicked")
		playback_speed = max(playback_speed / 1.5, 1.0)
	)

	_btn_mode.pressed.connect(func():
		print(">> MODE clicked")
		_pan_mode = !_pan_mode
		if _pan_mode:
			_btn_mode.text     = "🖱 PAN mode"
			_btn_mode.modulate = Color(1.0, 0.65, 0.1)
		else:
			_btn_mode.text     = "🖱 ORBIT mode"
			_btn_mode.modulate = Color(1.0, 1.0, 1.0)
	)


# =============================================================================
# Camera
# =============================================================================

func _apply_camera():
	if _camera == null or _pivot == null:
		return
	var az = deg_to_rad(_cam_azimuth)
	var el = deg_to_rad(_cam_elev)
	var d  = _cam_dist
	_camera.position = Vector3(
		d * cos(el) * sin(az),
		d * sin(el),
		d * cos(el) * cos(az)
	)
	_camera.look_at(_pivot.global_position)


func _pan_mouse(screen_delta: Vector2):
	if _camera == null or _pivot == null:
		return
	var spd   = _cam_dist * 0.012
	var right = _camera.global_transform.basis.x
	var fwd   = -_camera.global_transform.basis.z
	right.y   = 0.0
	fwd.y     = 0.0
	if right.length() > 0.001: right = right.normalized()
	if fwd.length()   > 0.001: fwd   = fwd.normalized()
	_pivot.position -= right * screen_delta.x * spd
	_pivot.position += fwd  * screen_delta.y * spd


# =============================================================================
# Load replay.json
# =============================================================================

func _load_replay():
	if not FileAccess.file_exists(replay_path):
		push_error("replay.json not found: " + replay_path)
		_label_tick.text = "ERROR: replay.json not found — see Output panel"
		return
	var file   = FileAccess.open(replay_path, FileAccess.READ)
	var parsed = JSON.parse_string(file.get_as_text())
	file.close()
	if parsed == null:
		push_error("Failed to parse replay.json")
		return
	_meta   = parsed["meta"]
	_layout = parsed["layout"]
	_frames = parsed["frames"]
	print("Replay loaded: %d frames | %d agents | %dx%d" % [
		_frames.size(), int(_meta["n_agents"]),
		int(_meta["grid_rows"]), int(_meta["grid_cols"])])


# =============================================================================
# Build 3D scene
# =============================================================================

func _build_scene():
	var grid_rows : int   = int(_layout["grid_rows"])
	var grid_cols : int   = int(_layout["grid_cols"])
	var cx        : float = grid_cols * cell_size_m * 0.5
	var cz        : float = grid_rows * cell_size_m * 0.5

	# Floor
	var floor_node     = MeshInstance3D.new()
	var plane          = PlaneMesh.new()
	plane.size         = Vector2(grid_cols * cell_size_m, grid_rows * cell_size_m)
	floor_node.mesh    = plane
	floor_node.position = Vector3(cx, 0.0, cz)
	var floor_mat      = StandardMaterial3D.new()
	floor_mat.albedo_color = Color(0.88, 0.90, 0.94)
	floor_node.set_surface_override_material(0, floor_mat)
	add_child(floor_node)

	# Grid lines
	var line_mat = StandardMaterial3D.new()
	line_mat.albedo_color = Color(0.68, 0.70, 0.76)
	for r in range(grid_rows + 1):
		var ln = MeshInstance3D.new()
		var bm = BoxMesh.new()
		bm.size = Vector3(grid_cols * cell_size_m, 0.02, 0.04)
		ln.mesh = bm
		ln.position = Vector3(cx, 0.02, r * cell_size_m)
		ln.set_surface_override_material(0, line_mat)
		add_child(ln)
	for c in range(grid_cols + 1):
		var ln = MeshInstance3D.new()
		var bm = BoxMesh.new()
		bm.size = Vector3(0.04, 0.02, grid_rows * cell_size_m)
		ln.mesh = bm
		ln.position = Vector3(c * cell_size_m, 0.02, cz)
		ln.set_surface_override_material(0, line_mat)
		add_child(ln)

	# Shelves — pallet racks (blue uprights, orange beams, brown cardboard pallets)
	for cell in _layout["shelves"]:
		_build_pallet_rack(int(cell[0]), int(cell[1]))

	# Depot pads
	var depot_mat = StandardMaterial3D.new()
	depot_mat.albedo_color     = Color(0.12, 0.38, 0.80)
	depot_mat.emission_enabled = true
	depot_mat.emission         = Color(0.05, 0.18, 0.50)
	for cell in _layout["depots"]:
		var pad = MeshInstance3D.new()
		var bm  = BoxMesh.new()
		bm.size = Vector3(cell_size_m * 0.92, 0.10, cell_size_m * 0.92)
		pad.mesh = bm
		pad.position = Vector3((int(cell[1]) + 0.5) * cell_size_m, 0.05,
							   (int(cell[0]) + 0.5) * cell_size_m)
		pad.set_surface_override_material(0, depot_mat)
		add_child(pad)

	# Agent workers (Tesco navy shirt + hard hat + hi-vis state vest)
	for i in range(int(_layout["n_agents"])):
		var worker_root = Node3D.new()
		var vest_mat    = _build_worker(worker_root)
		add_child(worker_root)
		_agent_nodes.append(worker_root)
		_agent_mats.append(vest_mat)

	# Pivot + camera (spherical coordinates)
	_pivot          = Node3D.new()
	_pivot.position = Vector3(cx, 0.0, cz)
	add_child(_pivot)
	_camera = Camera3D.new()
	_pivot.add_child(_camera)
	_cam_dist = max(grid_rows, grid_cols) * 0.75
	_apply_camera()

	# Lighting
	var light              = DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-50, 25, 0)
	light.light_energy     = 0.5
	light.shadow_enabled   = true
	add_child(light)

	var env_node             = WorldEnvironment.new()
	var env                  = Environment.new()
	env.background_mode      = Environment.BG_COLOR
	env.background_color     = Color(0.08, 0.10, 0.14)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color  = Color(0.62, 0.66, 0.75)
	env.ambient_light_energy = 0.85
	env_node.environment     = env
	add_child(env_node)


# =============================================================================
# Per-frame updates
# =============================================================================

func _update_agents(tick_idx: int):
	if tick_idx >= _frames.size():
		return
	for agent_data in _frames[tick_idx]["agents"]:
		var i : int = agent_data["id"]
		if i >= _agent_nodes.size():
			continue
		_agent_nodes[i].position = Vector3(
			float(agent_data["x"]), 0.0, float(agent_data["y"]))
		var col : Color
		match agent_data["state"]:
			"waiting":  col = COLOR_WAITING
			"to_item":  col = COLOR_TO_ITEM
			"picking":  col = COLOR_PICKING
			"to_depot": col = COLOR_TO_DEPOT
			"resting":  col = COLOR_RESTING
			"done":     col = COLOR_DONE
			_:          col = COLOR_WAITING
		_agent_mats[i].albedo_color = col
		_agent_mats[i].emission     = col * 0.35


func _update_hud(tick_idx: int):
	if _label_tick == null:
		return
	var total : int = _frames.size()
	var pct   : int = int(float(tick_idx) / float(max(total, 1)) * 100.0)
	var hrs   : int = tick_idx / 3600
	var mins  : int = (tick_idx % 3600) / 60
	var secs  : int = tick_idx % 60
	_label_tick.text = (
		"Tick %d/%d (%d%%)  |  %dh %02dm %02ds  |  %.0fx  |  %s"
		% [tick_idx, total, pct, hrs, mins, secs, playback_speed,
		   "⏸ PAUSED" if _paused else "▶ PLAYING"]
	)
	if _label_info:
		_label_info.text = (
			"[L-drag] orbit  |  [Scroll] zoom  |  Click 'ORBIT mode' to switch to pan"
		)


# =============================================================================
# Pallet rack builder — blue uprights, orange beams, brown cardboard pallets
# =============================================================================

const RACK_LEVELS      : int   = 3
const RACK_HEIGHT      : float = 3.6
const UPRIGHT_THICK    : float = 0.08
const BEAM_THICK       : float = 0.10
const PALLET_INSET     : float = 0.08

const COLOR_UPRIGHT    = Color(0.10, 0.22, 0.55)   # rack blue
const COLOR_BEAM       = Color(0.95, 0.45, 0.10)   # rack orange
const COLOR_CARDBOARD  = Color(0.62, 0.43, 0.27)   # kraft brown

func _make_box(size: Vector3, pos: Vector3, mat: StandardMaterial3D) -> MeshInstance3D:
	var mi = MeshInstance3D.new()
	var bm = BoxMesh.new()
	bm.size = size
	mi.mesh = bm
	mi.position = pos
	mi.set_surface_override_material(0, mat)
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	add_child(mi)
	return mi


func _build_pallet_rack(row: int, col: int):
	var cx : float = (col + 0.5) * cell_size_m
	var cz : float = (row + 0.5) * cell_size_m
	var w  : float = cell_size_m
	var d  : float = cell_size_m

	var up_mat = StandardMaterial3D.new()
	up_mat.albedo_color = COLOR_UPRIGHT
	up_mat.metallic = 0.3
	up_mat.roughness = 0.55

	var beam_mat = StandardMaterial3D.new()
	beam_mat.albedo_color = COLOR_BEAM
	beam_mat.metallic = 0.3
	beam_mat.roughness = 0.55

	var card_mat = StandardMaterial3D.new()
	card_mat.albedo_color = COLOR_CARDBOARD
	card_mat.roughness = 0.95

	# 4 vertical uprights at corners of the cell
	var ox = (w * 0.5) - UPRIGHT_THICK * 0.5
	var oz = (d * 0.5) - UPRIGHT_THICK * 0.5
	for sx in [-1, 1]:
		for sz in [-1, 1]:
			_make_box(
				Vector3(UPRIGHT_THICK, RACK_HEIGHT, UPRIGHT_THICK),
				Vector3(cx + sx * ox, RACK_HEIGHT * 0.5, cz + sz * oz),
				up_mat)

	# Horizontal beams along the long (col) axis — one set per level
	var level_h = RACK_HEIGHT / float(RACK_LEVELS)
	for lv in range(RACK_LEVELS):
		var y_beam = level_h * lv + 0.02
		# front + back beams (run along x)
		for sz in [-1, 1]:
			_make_box(
				Vector3(w - UPRIGHT_THICK, BEAM_THICK, BEAM_THICK),
				Vector3(cx, y_beam + BEAM_THICK * 0.5, cz + sz * oz),
				beam_mat)
		# A subtle deck plank for pallets to rest on
		var deck_y = y_beam + BEAM_THICK
		# Stack pallet boxes on this level (1-2 cardboard cartons, varied size)
		var n_boxes = 1 + (lv + row + col) % 2
		for bi in range(n_boxes):
			var box_w = (w - PALLET_INSET * 2.0) / float(n_boxes) - 0.06
			var box_h = 0.55 + 0.15 * float((bi + lv + col) % 3)
			var box_d = d - PALLET_INSET * 2.0 - 0.08
			var bx = cx - (w * 0.5) + PALLET_INSET + box_w * 0.5 + bi * (box_w + 0.06)
			_make_box(
				Vector3(box_w, box_h, box_d),
				Vector3(bx, deck_y + box_h * 0.5, cz),
				card_mat)


# =============================================================================
# Worker builder — Tesco navy shirt + hard hat + hi-vis state vest
# Returns the vest material so per-frame state colours can be applied.
# =============================================================================

const COLOR_SHIRT   = Color(0.05, 0.13, 0.36)   # Tesco navy
const COLOR_TROUSER = Color(0.12, 0.13, 0.16)
const COLOR_SKIN    = Color(0.92, 0.76, 0.62)
const COLOR_HAT     = Color(1.00, 0.62, 0.05)   # safety orange hard hat
const COLOR_BOOT    = Color(0.06, 0.06, 0.07)

func _add_child_mesh(parent: Node3D, mesh: Mesh, pos: Vector3, mat: StandardMaterial3D) -> MeshInstance3D:
	var mi = MeshInstance3D.new()
	mi.mesh = mesh
	mi.position = pos
	mi.set_surface_override_material(0, mat)
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	parent.add_child(mi)
	return mi


func _build_worker(root: Node3D) -> StandardMaterial3D:
	var shirt_mat   = StandardMaterial3D.new(); shirt_mat.albedo_color   = COLOR_SHIRT;   shirt_mat.roughness = 0.85
	var trouser_mat = StandardMaterial3D.new(); trouser_mat.albedo_color = COLOR_TROUSER; trouser_mat.roughness = 0.90
	var skin_mat    = StandardMaterial3D.new(); skin_mat.albedo_color    = COLOR_SKIN;    skin_mat.roughness = 0.75
	var hat_mat     = StandardMaterial3D.new(); hat_mat.albedo_color     = COLOR_HAT;     hat_mat.roughness = 0.40
	var boot_mat    = StandardMaterial3D.new(); boot_mat.albedo_color    = COLOR_BOOT;    boot_mat.roughness = 0.60

	# Hi-vis vest material — colour driven by agent state (returned to caller).
	var vest_mat = StandardMaterial3D.new()
	vest_mat.albedo_color     = COLOR_WAITING
	vest_mat.emission_enabled = true
	vest_mat.emission         = COLOR_WAITING * 0.35
	vest_mat.roughness        = 0.45

	# Legs (two cylinders)
	var leg_mesh = CylinderMesh.new()
	leg_mesh.top_radius = 0.09
	leg_mesh.bottom_radius = 0.10
	leg_mesh.height = 0.85
	_add_child_mesh(root, leg_mesh, Vector3(-0.10, 0.425, 0.0), trouser_mat)
	_add_child_mesh(root, leg_mesh, Vector3( 0.10, 0.425, 0.0), trouser_mat)

	# Boots
	var boot_mesh = BoxMesh.new()
	boot_mesh.size = Vector3(0.18, 0.10, 0.28)
	_add_child_mesh(root, boot_mesh, Vector3(-0.10, 0.05, 0.04), boot_mat)
	_add_child_mesh(root, boot_mesh, Vector3( 0.10, 0.05, 0.04), boot_mat)

	# Torso — navy shirt
	var torso_mesh = BoxMesh.new()
	torso_mesh.size = Vector3(0.46, 0.60, 0.28)
	_add_child_mesh(root, torso_mesh, Vector3(0.0, 1.15, 0.0), shirt_mat)

	# Hi-vis vest — slightly larger and shorter than torso, sits on top of shirt
	var vest_mesh = BoxMesh.new()
	vest_mesh.size = Vector3(0.50, 0.48, 0.31)
	_add_child_mesh(root, vest_mesh, Vector3(0.0, 1.18, 0.0), vest_mat)

	# Arms — short cylinders along the torso sides
	var arm_mesh = CylinderMesh.new()
	arm_mesh.top_radius = 0.07
	arm_mesh.bottom_radius = 0.07
	arm_mesh.height = 0.55
	_add_child_mesh(root, arm_mesh, Vector3(-0.30, 1.18, 0.0), shirt_mat)
	_add_child_mesh(root, arm_mesh, Vector3( 0.30, 1.18, 0.0), shirt_mat)

	# Head
	var head_mesh = SphereMesh.new()
	head_mesh.radius = 0.13
	head_mesh.height = 0.26
	_add_child_mesh(root, head_mesh, Vector3(0.0, 1.62, 0.0), skin_mat)

	# Hard hat — short cylinder on top, plus a thin brim
	var hat_mesh = CylinderMesh.new()
	hat_mesh.top_radius = 0.13
	hat_mesh.bottom_radius = 0.15
	hat_mesh.height = 0.13
	_add_child_mesh(root, hat_mesh, Vector3(0.0, 1.79, 0.0), hat_mat)
	var brim_mesh = CylinderMesh.new()
	brim_mesh.top_radius = 0.20
	brim_mesh.bottom_radius = 0.20
	brim_mesh.height = 0.02
	_add_child_mesh(root, brim_mesh, Vector3(0.0, 1.73, 0.03), hat_mat)

	return vest_mat

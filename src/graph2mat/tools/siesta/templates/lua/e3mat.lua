{% if include_store_code %}
local store_dir = "{{ store_dir }}"
local store_interval = {{ store_interval }}
local store_step_prefix = "{{ store_step_prefix }}"
local store_files = "{{ store_files }}"

local istep_store = 0

{% endif %}
{% if include_server_code %}
local server_address = "{{ server_address }}"
local history_len = {{ history_len }}
local main_fdf = "{{ main_fdf }}"
local matrix_ref = "{{ ml_model_name }}"
local work_dir = {{ work_dir }}

local main_fdf_path = work_dir .. "/" .. main_fdf
local istep_extrapolation = 0

{% endif %}
function siesta_comm()

    {% if include_store_code %}
    -- Initialize the storage directory if this is the beggining of the MD
    if siesta.state == siesta.INIT_MD then
        init_store_dir()
    end

    -- After each step, store the step files
    if siesta.state == siesta.FORCES then
        store_step(istep_store)
        istep_store = istep_store + 1
    end
    {% endif %}

    {% if include_server_code %}
    -- On initialization, write the prediction for the starting structure
    if siesta.state == siesta.INITIALIZE then
        init_history(history_len, matrix_ref)
    end

    if siesta.state == siesta.INIT_MD then
        if istep_extrapolation == 0 then
        add_step(main_fdf_path, matrix_ref)
        end
    end

    if siesta.state == siesta.FORCES then
        add_matrix(main_fdf_path)
    end

    if siesta.state == siesta.AFTER_MOVE then
        istep_extrapolation = istep_extrapolation + 1

        add_step(main_fdf_path, matrix_ref)
        predict_next(work_dir .. "/siesta.DM")

    end
    {% endif %}

end

{% if include_store_code %}
-- ----------------------------------------------------
--           MD STORAGE HELPER FUNCTIONS
-- ----------------------------------------------------

function init_store_dir()

    if not siesta.IONode then
        -- only allow the IOnode to perform stuff...
        return
    end

    -- Create the directory where the dataset will be stored
    os.execute("mkdir " .. store_dir)

    -- Store the basis
    os.execute("mkdir " ..  store_dir .. "/basis")
    os.execute("cp *.ion* " .. store_dir .. "/basis")
end

function store_step(istep)

    if not siesta.IONode then
        -- only allow the IOnode to perform stuff...
        return
    end

    -- If the step is a multiple of the store interval, store the frame
    if istep % store_interval == 0 then
        os.execute("mkdir " .. store_dir .. "/" .. store_step_prefix .. istep)
        os.execute("cp " .. store_files .. " " .. store_dir .. "/" .. store_step_prefix .. istep)
    end


end
{% endif %}

{% if include_server_code %}
-- ----------------------------------------------------
--           SERVER FUNCTIONALITY
-- ----------------------------------------------------

function server_get(path)
    if not siesta.IONode then
        -- only IO node communicates with the server
        return
    end

    os.execute("curl '" .. server_address .. "/" .. path .. "'")
 end

 function init_history(len, matrix_ref)
    server_get("init?history_len=" .. len .. "&data_processor=" .. matrix_ref)
 end

 function add_step(path, matrix_ref)
    server_get("add_step?path=" .. path .. "&matrix_ref=" .. matrix_ref)
 end

 function add_matrix(path)
    server_get("add_matrix?path=" .. path)
 end

 function setup_processor(basis_dir, matrix)
    server_get("setup_processor?basis_dir=" .. basis_dir .. "&matrix=" .. matrix)
 end

 function predict_next(out)
    server_get("extrapolate?out=" .. out)
 end

{% endif %}

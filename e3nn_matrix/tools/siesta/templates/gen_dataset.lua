local istep = 0

function siesta_comm()

    -- Do the actual communication with SIESTA
    if siesta.state == siesta.INIT_MD then
        init_dir()
    end
    -- Do the actual communication with SIESTA
    if siesta.state == siesta.FORCES then
        store_frame(istep)
        istep = istep + 1
    end

end

function init_dir()

    if not siesta.IONode then
        -- only allow the IOnode to perform stuff...
        return
    end

    -- Create the directory where the dataset will be stored
    os.execute("mkdir {{ dataset_dir }}")

    -- Store the basis
    os.execute("mkdir {{ dataset_dir }}/basis")
    os.execute("cp *.ion* {{ dataset_dir }}/basis")
end

function store_frame(istep)

    if not siesta.IONode then
        -- only allow the IOnode to perform stuff...
        return
    end

    -- If the step is a multiple of the store interval, store the frame
    if istep % {{ store_interval }} == 0 then
        os.execute("mkdir {{ dataset_dir }}/{{ stepdir_prefix }}" .. istep)
        os.execute("cp {{ files_to_keep }} {{ dataset_dir }}/{{ stepdir_prefix }}" .. istep)
    end

    
end

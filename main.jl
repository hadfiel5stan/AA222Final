using FileIO
using MeshIO
using Meshes
using GLMakie
using GeometryBasics
using Distributions
using LinearAlgebra
using Evolutionary
using Plots

struct Problem
    breaking_points
    veins
    lvs
    diameters
    leakages
    flowrates
end

function flow_leaked(solns, p)
    ret = 0

    # Either we've got
    for (i, v) in enumerate(solns)
        # No Anastomosis, leak the flow rate times the leakage percentage
        if isnothing(solns[i][1])
            ret += p.flowrates[i]*p.leakages[i]
        # Integrate the cdf to figure out the probability of success, expected flow leaked
        # is the flowrate*(1 - prob_success) as full flow will be leaked if unsuccessful
        else
            psuccess = cdf(p.breaking_points, p.diameters[i])
            ret += p.flowrates[i]*(1-psuccess)
        end
    end

    return ret
end

function evaluatesoln(soln, p, bbox)
    # Soln will be of form [[a1, b1],[a2, b2],...]
    # Where
    ### a_n is the location of the sever on the Lymph vessel
    ### b_n is the recipient location in the veins

    flowleak = 0;

    # We will use the success probability distribution to the expected diverted flow
    flowleak = flow_leaked(soln, p)

    area = abs(bbox[1][1] - bbox[1][2])*abs(bbox[2][1] - bbox[2][2])

    diameterweightedd = 0
    for (i, v) in enumerate(soln)
        if !isnothing(v[1])
            diameterweightedd += norm(v[1] - v[2])/p.diameters[i]
        end
    end

    return flowleak, area, diameterweightedd
end

function bbox_from_soln(soln)
    # bb is [xbounds, zbounds]
    bb = [[Inf, -Inf],[Inf, -Inf]]
    for (i, v) in enumerate(soln)
        lpoint = v[1]
        vpoint = v[2]

        if isnothing(lpoint)
            continue
        end

        d = norm(vpoint - lpoint)

        # Since we'll need to open up enough to stretch the lymph, find the lower and upper ends of
        # incision necessary
        low = lpoint - [0, 0, d];
        high = lpoint + [0, 0, d];
        
        # update the bounding box
        for pt in [lpoint, vpoint, low, high]
            updatebb!(bb, pt[1], pt[3])
        end
    end
    return bb
end

function create_soln_from_guess(guess, p; threed=true)
    # Guess will be of form [a1, a2, ... an]
    # where a_i is the z location to sever on lymph i 

    # bb is [xbounds, zbounds]
    bb = [[Inf, -Inf],[Inf, -Inf]]
    soln = [[] for a in guess]

    guess = 40*guess;

    for (i, v) in enumerate(guess)
        if v < 0 || v > 40
            soln[i] = [nothing, nothing]
        else
            # Find closest point on lymph to that z location
            lpoint = find_closest_lymph(p.lvs[i], guess[i])

            # Find the closest point on the viens around
            if threed
                vpoint = closest_vein(p, lpoint)
            else
                vpoint = closest_vein_2d(p, lpoint)
            end

            # Distance between them
            d = norm(vpoint - lpoint);

            # updates solution
            soln[i] = [lpoint, vpoint];

            # Since we'll need to open up enough to stretch the lymph, find the lower and upper ends of
            # incision necessary
            low = lpoint - [0, 0, d];
            high = lpoint + [0, 0, d];

            # update the bounding box
            for pt in [lpoint, vpoint, low, high]
                updatebb!(bb, pt[1], pt[3])
            end
        end
    end

    if bb[1][1] == Inf
        bb == [[0,0],[0,0]]
    end

    return soln, bb
end

function updatebb!(bb, x, y)
    if x < bb[1][1]
        bb[1][1] = x
    end
    if x > bb[1][2]
        bb[1][2] = x
    end
    if y < bb[2][1]
        bb[2][1] = y
    end
    if y > bb[2][2]
        bb[2][2] = y
    end
    return
end

function find_closest_lymph(lv, z)
    # Function takes a z location and returns the point that is closest to that z coordinate
    ds = zeros(length(coordinates(lv)), 1)

    for (i, c) in enumerate(GeometryBasics.coordinates(lv))
        ds[i] = abs(c[3] - z)
    end

    closest = minimum(ds)
    
    indices = findall(x->x==closest, ds);

    points = [coordinates(lv)[i] for i in indices]

    return mean(points) 
end

function generate_bps(l)
    n = Distributions.Normal(0.45, 0.05);

    breakingpoints = abs.(rand(n, l))

    return breakingpoints
end

function closest_vein(p, x)
    ds = zeros(length(GeometryBasics.coordinates(p.veins)), 1)
    
    for (i, c) in enumerate(GeometryBasics.coordinates(p.veins))
        ds[i] = norm(c - x)
    end

    j = argmin(ds)
    return GeometryBasics.coordinates(p.veins)[j]
end

function closest_vein_2d(p, x)
    ds = zeros(length(GeometryBasics.coordinates(p.veins)), 1)
    x2d = [x[1], x[3]]

    for (i, c) in enumerate(GeometryBasics.coordinates(p.veins))
        c2d = [c[1], c[3]]
        ds[i] = norm(c2d - x2d)
    end

    j = argmin(ds)
    return GeometryBasics.coordinates(p.veins)[j]
end

function initialize_problem(vpath, paths, ds, cutoffdia=0.5)
    thispath = @__DIR__

    modelpath = split(thispath, "/")
    modelpath = modelpath[1:end-1]
    push!(modelpath, "Models")

    veins = load(vpath)

    lvs = []

    for i in eachindex(paths)
        push!(lvs, load(paths[i]))
    end

    breakingpoints = Distributions.Normal(cutoffdia, 0.05);

    porosities = [0.8, 0.5, 0.5, 0.3]

    p = Problem(breakingpoints, veins, lvs, ds, porosities, ones(length(lvs), 1))

    return p
end

function mooproblem(x, p)

    s, bb = create_soln_from_guess(x, p)
    t1, t2, t3 = evaluatesoln(s, p, bb)
    return [t1, t2]
end

function sooproblem(x, p; priority = 2, threed=true)
    s, bb = create_soln_from_guess(x, p, threed=threed)
    results = evaluatesoln(s, p, bb)
    # In the single objective case, we only return the priority
    return results[priority]
end

function cross_entropy_method(f, P, k_max; m=100, m_elite=10) 
    Ps = []
    
    for k in 1 : k_max
        println("starting generation"*string(k))

        samples = rand(P, m)
        # Have a very specialized constraint that we need all guesses to be within 0-1
        for i in eachindex(samples[1, :])
            t = false
            while any(samples[:, i] .< 0) || any(samples[:, i] .> 1)
                t = true
                samples[:, i] = rand(P, 1)
            end
            if t
                # println(samples[:, i])
            end
        end
        # println("Fixed samples from generation "*string(k))

        order = sortperm([f(samples[:,i]) for i in 1:m])

        println("Evaluated Samples")
        P = fit(typeof(P), samples[:,order[1:m_elite]])
        push!(Ps, P)
    end
    return P, Ps
end

function generate_pareto_fronteir(p; threed=true)
    # Uses X-Entropy to generate the pareto fronteir since there are a finite number of cases for
    # the first objective
    
    solguesses = []
    solvals = []

    fs = [x->sooproblem([x[1], x[2], x[3], x[4]], p, threed=threed),
            x->sooproblem([-1, x[1], x[2], x[3]], p, threed=threed),
            x->sooproblem([x[1], -1, x[2], x[3]], p, threed=threed),
            x->sooproblem([x[1], x[2], -1, x[3]], p, threed=threed),
            x->sooproblem([x[1], x[2], x[3], -1], p, threed=threed),
            x->sooproblem([-1, -1, x[1], x[2]], p, threed=threed),
            x->sooproblem([-1, x[1], -1, x[2]], p, threed=threed),
            x->sooproblem([-1, x[1], x[2], -1], p, threed=threed),
            x->sooproblem([x[1], -1, x[2], -1], p, threed=threed),
            x->sooproblem([x[1], x[2], -1, -1], p, threed=threed),
            x->sooproblem([x[1], -1, -1, x[2]], p, threed=threed),
            x->sooproblem([x[1], -1, -1, -1], p, threed=threed),
            x->sooproblem([-1, x[1], -1, -1], p, threed=threed),
            x->sooproblem([-1, -1, x[1], -1], p, threed=threed),
            x->sooproblem([-1, -1, -1, x[1]], p, threed=threed)]


    Σ4 = 0.01*ones(4, 4) + 0.24*I
    μ4 = [0.5;0.5;0.5;0.5];       
    Σ3 = 0.01*ones(3, 3) + 0.24*I
    μ3 = [0.5;0.5;0.5];
    Σ2 = 0.01*ones(2, 2) + 0.24*I
    μ2 = [0.5;0.5];
    Σ1 = [0.25];
    μ1 = [0.5];

    for (i, v) in enumerate(fs)
        if i == 1
            N = Distributions.MvNormal(μ4, Σ4)
        elseif i < 6
            N = Distributions.MvNormal(μ3, Σ3)
        elseif i < 12
            N = Distributions.MvNormal(μ2, Σ2)
        else
            N = Distributions.MvNormal(μ1, Σ1)
        end

        result, history = cross_entropy_method(v, N, 10)
        resultguess = rebuild_guess(mean(result), i)
        
        push!(solguesses, resultguess)
        push!(solvals, mooproblem(resultguess, p))

        println("Completed Generation "*string(i))

    end
    
    return solvals, solguesses
end

function rebuild_guess(x, case)
    # Bush league, but reconstruct the full guess vector from the reduced vector and an identifier
    # for which case
    if case == 1
        return x
    elseif case == 2
        return [-1, x[1], x[2], x[3]]
    elseif case == 3
        return [x[1], -1, x[2], x[3]]
    elseif case == 4
        return [x[1], x[2], -1, x[3]]
    elseif case == 5
        return [x[1], x[2], x[3], -1]
    elseif case == 6
        return [-1, -1, x[1], x[2]]
    elseif case == 7
        return [-1, x[1], -1, x[2]]
    elseif case == 8
        return [-1, x[1], x[2], -1]
    elseif case == 9
        return [x[1], -1, x[2], -1]
    elseif case == 10
        return [x[1], x[2], -1, -1]
    elseif case == 11
        return [x[1], -1, -1, x[2]]
    elseif case == 12
        return [x[1], -1, -1, -1]
    elseif case == 13
        return [-1, x[1], -1, -1]
    elseif case == 14
        return [-1, -1, x[1], -1]
    elseif case == 15
        return [-1, -1, -1, x[1]]
    end
end

function mydominated(a, b)
   # Returns true if a is dominated by b
   # returns false otherwise 

   return all(b .< a)
end

function get_dominators(set)
    pareto_front = []
    others = []

    for (i, s) in enumerate(set)
        pareto = true
        for (j, q) in enumerate(set)
            if i != j && mydominated(s, q)
                pareto = false
                break
            end
        end
        if pareto
            push!(pareto_front, s)
        else
            push!(others, s)
        end
    end

    return pareto_front, others
end

function plot_convergence(case, p; threed=true)
    fs = [x->sooproblem([x[1], x[2], x[3], x[4]], p, threed=threed),
    x->sooproblem([-1, x[1], x[2], x[3]], p, threed=threed),
    x->sooproblem([x[1], -1, x[2], x[3]], p, threed=threed),
    x->sooproblem([x[1], x[2], -1, x[3]], p, threed=threed),
    x->sooproblem([x[1], x[2], x[3], -1], p, threed=threed),
    x->sooproblem([-1, -1, x[1], x[2]], p, threed=threed),
    x->sooproblem([-1, x[1], -1, x[2]], p, threed=threed),
    x->sooproblem([-1, x[1], x[2], -1], p, threed=threed),
    x->sooproblem([x[1], -1, x[2], -1], p, threed=threed),
    x->sooproblem([x[1], x[2], -1, -1], p, threed=threed),
    x->sooproblem([x[1], -1, -1, x[2]], p, threed=threed),
    x->sooproblem([x[1], -1, -1, -1], p, threed=threed),
    x->sooproblem([-1, x[1], -1, -1], p, threed=threed),
    x->sooproblem([-1, -1, x[1], -1], p, threed=threed),
    x->sooproblem([-1, -1, -1, x[1]], p, threed=threed)]


    Σ4 = 0.01*ones(4, 4) + 0.24*I
    μ4 = [0.5;0.5;0.5;0.5];       
    Σ3 = 0.01*ones(3, 3) + 0.24*I
    μ3 = [0.5;0.5;0.5];
    Σ2 = 0.01*ones(2, 2) + 0.24*I
    μ2 = [0.5;0.5];
    Σ1 = [0.25];
    μ1 = [0.5];

    if case == 1
        N = Distributions.MvNormal(μ4, Σ4)
    elseif case < 6
        N = Distributions.MvNormal(μ3, Σ3)
    elseif case < 12
        N = Distributions.MvNormal(μ2, Σ2)
    else
        N = Distributions.MvNormal(μ1, Σ1)
    end

    result, history = cross_entropy_method(fs[case], N, 10)
    resultguesses = [rebuild_guess(mean(r), case) for r in history]

    resultvals = [mooproblem(rg, p)[2] for rg in resultguesses]

    #return Plots.plot(resultvals, xlabel="Iteration", ylabel="Area Opened During Surgery", title="Convergence of Pareto Optimal Solution", legend=false)
    return resultguesses
end

function in2mm(x)
    return x*25.4
end

function plotsolution(s, p)
    f = Makie.mesh(p.veins)
    for lv in p.lvs
        f = Makie.mesh!(lv)
    end
    
    display(f)
end

veins = "/Users/brandonhadfield/Documents/School/SPR2023/AA222/FinalProject/Models/LVA_VE.stl"

lv1 = "/Users/brandonhadfield/Documents/School/SPR2023/AA222/FinalProject/Models/LV1.stl"
lv2 = "/Users/brandonhadfield/Documents/School/SPR2023/AA222/FinalProject/Models/LV2.stl"
lv3 = "/Users/brandonhadfield/Documents/School/SPR2023/AA222/FinalProject/Models/LV3.stl"
lv4 = "/Users/brandonhadfield/Documents/School/SPR2023/AA222/FinalProject/Models/LV4.stl"

p = initialize_problem(veins, [lv1, lv2, lv3, lv4], [0.5, 0.6, 0.8, 0.5])


# plot_convergence(8, p)
# add = 33.5
# addy = 1
# cguesses = [[[-15.97+add, 9.18+addy, 13.46], [-16.55+add, 17.24+addy, 11.3]], [[17.94+add, 30.07+addy, 24.90],[19.01+add, 30.41+addy, 25.33]], [[16.21+add, 11.73+addy, 30.0],[18.07+add, 21.46+addy, 30.0]], [nothing, nothing]]
# cbb = bbox_from_soln(cguesses)

# t1, t2, t3 = evaluatesoln(cguesses, p, cbb)

# println(t1)
# println(t2)

cguesses = plot_convergence(5, p)
cguesses = cguesses[end]
cguesses = create_soln_from_guess(cguesses, p)[1]

fig = Makie.mesh(p.veins, color=:blue)
for lv in p.lvs
    fig = Makie.mesh!(lv, color=:green)
end

for g in cguesses
    if isnothing(g[1])
        continue
    end
    p1 = GeometryBasics.Point{3, Float64}(g[1])
    p2 = GeometryBasics.Point{3, Float64}(g[2])
    cyl = GeometryBasics.Cylinder(p1, p2, 0.5)
    fig = Makie.mesh!(cyl, color=:red)
end

display(fig)

#solvals, solguesses = generate_pareto_fronteir(p, threed=false)

# solvals = [[1.022750132934767, 502.487464418482], [1.3227501329347668, 283.1003650427738], [1.5000000009865877, 500.21815591262566], [1.5227501319481793, 214.49200394721265], [0.8227501329347668, 502.487464418482], [1.8000000009865877, 283.3822600870408], [1.8227501319481791, 105.6874460230465], [1.1227501329347669, 25.39099473493843], [1.3000000009865877, 499.80682249124584], [1.3227501319481794, 107.15729318780359], [2.0, 21.93281106166978], [1.8, 0.685334250847518], [1.6227501319481792, 0.5462416491936892], [1.6000000009865878, 19.495511752407765], [2.3, 6.812851431350282]];
# solvals2d = [[1.022750132934767, 506.84805614688594], [1.3227501329347668, 322.5038370659895], [1.5000000009865877, 502.65601241839613], [1.5227501319481793, 215.0794430320675], [0.8227501329347668, 562.0891264723468], [1.8000000009865877, 329.2841410721303], [1.8227501319481791, 114.84979517542524], [1.1227501329347669, 54.13967256521573], [1.3000000009865877, 502.1212848072355], [1.3227501319481794, 107.15729318780359], [2.0, 21.93281106166978], [1.8, 12.28265380877383], [1.6227501319481792, 0.8149855186638888], [1.6000000009865878, 54.13967256521573], [2.3, 266.57290646723413]];
# csolval = [[t1, t2]]

# pfront, others = get_dominators(solvals)
# pfront2, others2 = get_dominators(solvals2d)

# sort!(pfront)
# sort!(pfront2)

# redux = zeros(length(pfront), 1)
# for (i, v) in enumerate(pfront)
#     redux[i] = (pfront2[i][2] - pfront[i][2])/pfront2[i][2]
# end

# println(csolval[1][2] - solvals[5][2])
# println((csolval[1][2] - solvals[5][2])/csolval[1][2])
# println(csolval[1][2] - solvals2d[5][2])
# println((csolval[1][2] - solvals2d[5][2])/csolval[1][2])

# Plots.scatter([s[1] for s in others], [s[2] for s in others], marker=:xcross, markercolor=:black, xlabel="Lymph Leaked", ylabel="Area Exposed in Surgery", title="Optimized Results", label="Dominated Solutions")
# Plots.scatter!([s[1] for s in others2], [s[2] for s in others2], marker=:xcross, markercolor=:black, xlabel="Lymph Leaked", ylabel="Area Exposed in Surgery", title="Optimized Results", label="")
# Plots.scatter([t1], [t2], marker=:xcross, markercolor=:black, xlabel="Lymph Leaked", ylabel="Area Exposed in Surgery", title="Human Solution Compared to Optimized Results", label="Human Solution")
# Plots.plot!([s[1] for s in pfront], [s[2] for s in pfront], marker=:xcross, color=:blue, linestyle=:dash, label="Pareto Frontier from 3D data")
# Plots.plot!([s[1] for s in pfront2], [s[2] for s in pfront2], marker=:xcross, color=:green, linestyle=:dash, label="Pareto Frontier from 2D data")

# f = x->sooproblem([-1, x[1], x[2], x[3]], p)

# Σ = 0.01*ones(3, 3) + 0.24*I
# μ = [0.5;0.5;0.5];

# N = Distributions.MvNormal(μ, Σ)

# result = cross_entropy_method(f, N, 10)

# s = mean(result)

# println(s)
# println(f(s))





# x0 = [0.5,0.5,0.5,0.5]
# F = f(x0)

# lb = [-1,-1,-1,-1]
# ub = [40,40,40,40]

# f = x->mooproblem(x, p)

# result = Evolutionary.optimize(f, F, x0, NSGA2(populationSize=50, 
#                     metrics=[Evolutionary.GD(1e-1, true)]), Evolutionary.Options(show_trace=true, store_trace=true, time_limit=1000.0))


# println(Evolutionary.minimizer(result))
# println(result)


# test = [[-0.6613291199354201, 0.35142296325504707, 1.169146441415488, 1.8476248137421385], [-0.3706385052747113, 0.5321997952823259, 1.0782737063161065, 1.7310114288256877], [-0.35499235472491925, 0.5303782944354553, 0.9916097815765055, 1.7582783570426128], [-0.9617241220801012, 0.15850296747457995, 1.2558888220136113, 1.8873718581846048], [-0.31239592659751214, 0.6023200448484459, 1.612965102181906, 1.8238746577012863], [-0.23537942512739735, 0.5180979853892317, 1.1525790181028097, 1.8003263733879091], [-0.726685394180458, 0.3079089892413072, 1.1332089379873929, 1.7756028143297713], [-0.5569067037309526, 0.2520791647441283, 1.1064560537832586, 1.6287037948442817], [-0.13201508260938277, 1.0783946765249166, 1.3473528614155426, 1.8587201230343227], [-0.5233646238923528, 0.7959661401565903, 1.1599663101358564, 2.2518643140538193], [-0.2433779108776038, 0.5137270046460206, 1.7812852757042321, 1.8008531488325867], [-0.2449220286158073, 0.8006904067945607, 1.562771869662002, 1.7549821496292042], [-1.199189415524989, 0.20131194998450516, 1.1775370283808215, 1.8493911281495228], [-0.31167604650838143, 0.44570106105785784, 0.9220493923227641, 1.8487263133708653], [-0.18766617055182183, 0.5195010872831655, 1.302972187187981, 1.7944598221644938], [-0.7126631911826407, 0.2008108355751415, 1.3895035591743388, 2.2972567765319223], [-0.3691487111126516, 0.2455308926460086, 0.8497004814774676, 1.8068519302019446], [-0.17843973897363877, 0.3732139674832908, 1.1439341590964816, 1.8655851906050647], [-0.2659185199258469, 0.5832211322395674, 1.576240075808103, 1.8119556408041202], [-0.24717690164291448, 0.5150469460130107, 1.2268843298895127, 1.7879045908012285], [-0.9970171100159118, 0.539200878536449, 1.0810837822870882, 1.7346962377090223], [-0.7307094324093881, 0.34134542601050527, 1.364871052614952, 1.6722599302319705], [-0.48071397019245543, 0.49132575435739484, 0.8873160389101652, 0.7273973744650449], [-0.7307094324093881, 0.3561830669964793, 1.7551934411764978, 1.3643549691516623], [-0.5878714315573936, 0.015648907296515857, 1.234828149641029, 2.0967507886419994], [-0.4719051112808899, 0.30332773182213457, 0.8601924855420948, 1.0921774804815385], [-0.4773120683850742, 0.16066755077086264, 1.3504639426992382, 1.847046981094734], [-0.35634574675623576, 0.3393827680800472, 1.3561633387188872, 1.8458850854586994], [-0.509105037369708, 0.7923856975295206, 1.0526780713418775, 2.0167370901448387], [-0.5019988814725946, 0.8480025465355818, 1.3681031929719798, 2.0319099774289695], [-0.052265588984100386, 0.5321954180616709, 0.997232560334485, 1.8578631236171077], [-0.6308981160635168, 0.34619856948588346, 2.2401296303430636, 1.8101738727442132], [-0.7325535145712949, 0.5260061632115183, 1.0466414714887133, 1.8169249881116256], [-0.39135139207486563, 0.5936926573078145, 1.3557086087445145, 1.851170798042628], [-0.3193674429364223, 0.5318889154288488, 1.1596471164895694, 1.6782148436017694], [-0.36157474276040474, 0.3579288907940761, 1.3081939445319257, 2.0425806836665754], [-0.2665618227949121, 0.6919557695170641, 1.5695059727350524, 1.8341766105004638], [-0.40204180268693357, 0.3835630553785695, 1.2380814640303588, 1.7083041928227551], [-0.2573175983288946, 0.7305863017537977, 1.6027061524984285, 1.6898573864849293], [-0.4802710117275522, 0.2197603997587965, 1.3530862757218762, 1.8491088895686811], [-0.3170069820409887, 0.8721746465927045, 1.3577280271937613, 1.945315050231646], [-0.11265604426732397, 0.543272225292071, 0.966071598626841, 1.8323575269095262], [-0.436318693373653, 0.6901011709089538, 1.2231884496440433, 1.8814607031273285], [-0.6502673113700171, 0.10822987143582868, 1.312165854407684, 2.2729466872570487], [-0.8505941484987918, 0.5246572482600435, 1.2728201106764008, 1.9406505527892461], [-0.37384856941555067, 0.3354815009884222, 1.2191901668466427, 1.8485279417506637], [-0.26597451601209016, 0.5830947277412004, 1.5728730242715776, 1.823066125652292], [-0.37624147412365594, 0.6530563510877706, 1.005587138904387, 1.9087951194887127], [-0.8375390358234971, 0.34205710392319955, 1.4119369864256535, 1.618918614199975], [-0.26701896600144165, 0.334769823075728, 1.1721242330359414, 1.901869257782659]]

# for t in test
#     println(mooproblem(t, p))
# end

# lv1 = 0.5 mm
# lv2 = 0.6 mm
# lv3 = 0.8 mm
# lv4 = 0.5 mm

#include "lenski_sim.hh"
#include <algorithm>
#include <numeric>
#include <errno.h>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <cassert>
#include <gsl/gsl_randist.h>
#include <boost/math/distributions/hypergeometric.hpp>


using std::normal_distribution;
using std::set_difference;


/* Class constructor. Initializes the simulation and sets 
 * the simulation parameters. */
lenski_sim::lenski_sim(
        const int _L,       
        const int _N_0, 
        const int _N_f,
        const double _p,    
        const double _rho,
        const double _sigh, 
        const double _muJ, 
        const double _sigJ,
        string _base_folder, 
        bool _interact,
        int _output_interval, 
        int _init_rank, 
        int _rank_interval,
        const int _nbins, 
        const double _min_select, 
        const double _max_select,  
        uint32_t _seed,
        const bool _hoc, 
        const int _replicate_number, 
        bool _output_sim_info,
        const int _reset_fac) : 

    /* Initialize the simulation parameters. */
    L(_L), N_0(_N_0), N_f(_N_f), p(_p*_L),
    rho(_rho), sigh(_sigh), sigJ(_sigJ), Foff(0), base_folder(_base_folder), 
    nbac_tot(_N_0), n_strains(1), interact(_interact), 
    output_interval(_output_interval), init_rank(_init_rank), 
    rank_interval(_rank_interval), bin_edges(new double[_nbins+1]), 
    bin_counters(new int[_nbins+2]), nbins(_nbins), 
    min_select(_min_select), max_select(_max_select), hoc(_hoc), 
    replicate_number(_replicate_number), 
    reset_index(1), reset_fac(_reset_fac)
    


    {
        // setup simulation space
        output_folder = base_folder + 
            "/replicate" + std::to_string(replicate_number);
        allocate_space(_seed, _output_sim_info);

        // draw random interactions and initializations
        setup_disorder();

        // set the initial rank
        if (init_rank >= 0) {  
            printf("About to update rank of the initial strain to %d.\n", 
                    init_rank); 
            update_rank();  
        }

        // output quenched disorder information to replicate folders
        if (_output_sim_info) {
            output_bin_edges();
            output_his_bin();
            if (interact) { output_Jijs_bin();  output_Jalpha_bin(); }
            output_alpha0s();
        }
    }


/* Scale what is needed to vary the strength of epistasis. */
void lenski_sim::scale_disorder(
        vector<double> unit_his, 
        vector<double> unit_Jijs) {

    // update the disorder values
    for (int ii = 0; ii < L; ii++) { his[ii] = sigh*unit_his[ii]; }
    int key(0);
    for (int ii = 0; ii < L; ii++) {
        for (int jj = ii+1; jj < L; jj++) {
            key = jj + ii*L; 
            Jijs[key] = sigJ*unit_Jijs[key];
        }
    }

    // after re-scaling h and J, the initialization cannot be the same, 
    // because the rescaling will have
    // changed the rank. hence, we need to re-compute the rank.
    // moreover, we need to update the values of J*alpha0 and Foff. 
    // this happens inside update_rank().
    if (init_rank >= 0) { 
        printf("About to update rank of the initial strain to %d.\n", 
                init_rank); 
        update_rank();  
    }

    else {
        update_Jalpha0(); 
        Foff = 0; 
        Foff = 1 - (hoc? compute_fitness_slow_hoc(0) 
                : compute_fitness_slow(0)); 
    }
}


/* Copy the quenched disorder. */
void lenski_sim::copy_disorder(
        vector<double> _his, 
        vector<double> _Jijs, 
        vector<double> _alpha0s, 
        vector<double> _Jalpha0) {

    his = _his;
    Jijs = _Jijs;

    // different initialization for each replicate.
    if (init_rank >= 0) {
        // for the new J matrix, draw a new uniform initialization, 
        // re-compute Jalpha0, and update to the correct rank.
        for (int ii = 0; ii < L; ii++) { 
            alpha0s[ii] = (gsl_rng_uniform(gsl_gen) < 0.5)? 1 : -1; 
        }

        update_Jalpha0();
        printf("About to update rank of the initial strain to %d.\n", 
                init_rank);
        update_rank();
    }

    // same initialization for each replicate.
    else {
        Jalpha0 = _Jalpha0;
        alpha0s = _alpha0s;
        Foff = 0; 
        Foff = 1 - (hoc? compute_fitness_slow_hoc(0) 
                : compute_fitness_slow(0));
    }

    mkdir(output_folder.c_str(), 0700);
    output_bin_edges();
    output_his_bin();
    if (interact) { output_Jijs_bin();  output_Jalpha_bin(); }
    output_alpha0s();
}


void lenski_sim::allocate_space(
        uint32_t seed, 
        bool output_sim_info) {

    // Declare the GSL random number generator
    // empirically, taus2 seems to be the fastest option.
    gsl_gen = gsl_rng_alloc(gsl_rng_taus2);  
    gsl_rng_set(gsl_gen, seed);

    // No mutations on the initial strain by definition.
    if (hoc) { mutations_hoc.emplace_back(); }
    else { mutations.emplace_back(); }
    mut_order.emplace_back();
    fit_effects.emplace_back();

    // We start with N_0 bacteria of the first strain by definition.
    n_bac.push_back(N_0);

    // the fitness of the first strain is one at the start by definition 
    // (choice of F_{offset}).
    fits.push_back(1);

    // And create the output folder.
    if (output_sim_info) { mkdir(output_folder.c_str(), 0700); }

    // Set up the bins and output information.
    setup_bins();

    // Preallocate space for the disorder.
    his.resize(L); alpha0s.resize(L);
    Jalpha0.resize(L, 0);
    Jijs.resize(L*L, 0);
}


/* Draw the quenched disorder. */
void lenski_sim::setup_disorder() {
    double curr_J_val(0);
    int key;

    for (int ii = 0; ii < L; ii++) {
        his[ii]     = gsl_ran_gaussian(gsl_gen, sigh);
        alpha0s[ii] = (gsl_rng_uniform(gsl_gen) < 0.5)? 1 : -1;
    }

    if (interact) {
        for (int ii = 0; ii < L; ii++) {
            for (int jj = ii+1; jj < L; jj++) {
                key = jj + ii*L;
                curr_J_val = (gsl_rng_uniform(gsl_gen) < rho)? 
                    gsl_ran_gaussian(gsl_gen, sigJ) : 0;

                if (curr_J_val != 0) { 
                    Jijs[key] = curr_J_val; 
                    Jalpha0[ii] += curr_J_val*alpha0s[jj];
                    Jalpha0[jj] += curr_J_val*alpha0s[ii];
                }
            }
        }
    }

    check_Jalpha0();
    Foff = 1 - (hoc? compute_fitness_slow_hoc(0) : compute_fitness_slow(0));
}


/* Draws a Poisson random number via inversion by sequential search. 
 * Fast for small values of $\mu$. */
int lenski_sim::draw_poisson(double lambda) {
    int x = 0;
    double p = exp(-lambda);
    double s = p;
    double u = gsl_rng_uniform(gsl_gen);
    while (u > s) { x++; p *= lambda/x; s += p; }
    return x;
}


/* Step the simulation forward by dt seconds. */
void lenski_sim::step_forward(double dt) {
    // Change in number of bacteria for the current (looped over) strain.
    double dN(0);         
    // Number of mutations for the current (looped over) strain in the time dt.
    int n_mutants(0),     
        new_strains(0), // Number of new strains after mutations have occurred.
        mutant_ind(0);  // Current index of the (looped over) mutation.
    double new_fit(0), new_effect(0);
    double alphak_p(0);
    double poisson_mu(0);

    // Compute the change in the number of cells for each strain, 
    // as well as the mutations that stem from each strain.
    for (int curr_strain = 0; curr_strain < n_strains; curr_strain++) {
        // figure out the number of mutant strains
        dN = n_bac[curr_strain]*(exp(fits[curr_strain]*dt) - 1);
        poisson_mu = dN*p;
        n_mutants = poisson_mu < 1.0? draw_poisson(poisson_mu) 
            : gsl_ran_poisson(gsl_gen, poisson_mu); // slightly speedier hack

        // ensure we don't draw more than the total population
        n_mutants = (n_mutants > dN)? dN : n_mutants; 
        n_mutants_so_far += n_mutants;

        // adjust the current and the overall population
        n_bac[curr_strain] += (dN - n_mutants);
        nbac_tot += dN;

        // for each mutant, flip the spin and join or define new strains.
        for (int curr_mutant = 0; curr_mutant < n_mutants; curr_mutant++) {
            // draw the random spin to flip
            mutant_ind = gsl_rng_uniform_int(gsl_gen, L); 

            // check if this strain already exists.
            auto curr_mut_order = mut_order[curr_strain];
            curr_mut_order.push_back(mutant_ind);
            auto mut_exists_it = current_strains.find(curr_mut_order);
            if (mut_exists_it != current_strains.cend()) {  
                n_bac[mut_exists_it->second]++;  
            }
            
            // otherwise, create a new strain
            else {
                // grab the parent set of mutations
                unordered_set<int> curr_muts = mutations[curr_strain];
                auto mut_iterator = curr_muts.find(mutant_ind);

                // update the set of mutations of the child
                if (mut_iterator == curr_muts.cend()) { 
                    curr_muts.insert(mutant_ind); 
                    alphak_p = alpha0s[mutant_ind]; 
                }

                else { 
                    curr_muts.erase(mut_iterator); 
                    alphak_p = -alpha0s[mutant_ind]; 
                }

                // compute the fitness using the parent set of mutations
                // note that curr_muts is fine to pass, because any 
                // additional term in compute_fitness will just be zero.
                new_fit = compute_fitness(curr_strain, mutant_ind, 
                                          curr_muts, alphak_p);  
                fits.push_back(new_fit);

                // update mutation information. 
                // avoid excessive copying with std::move
                mut_order.push_back(std::move(curr_mut_order));
                mutations.push_back(std::move(curr_muts));
                current_strains[curr_mut_order] = n_strains + new_strains;

                // store the fitness effect information
                auto curr_fit_effects = fit_effects[curr_strain];
                new_effect = new_fit - fits[curr_strain];
                curr_fit_effects.push_back(new_effect);
                fit_effects.push_back(std::move(curr_fit_effects));
                bin_counters[find_bin_ind(new_effect/fits[curr_strain])]++;

                // update strain counter and total number of bacteria
                new_strains++; 
                n_bac.push_back(1.0);
            }
        }
    }
    n_strains += new_strains;
}


/* Step the simulation forward by dt seconds, 
 * with an uncorrelated house of cards landscape at each gene. */
void lenski_sim::step_forward_hoc(double dt) {
    // Change in number of bacteria for the current (looped over) strain.
    double dN(0);         
    // Number of mutations for the current (looped over) strain in the time dt.
    int n_mutants(0),     
        new_strains(0), // Number of new strains after mutations have occurred.
        mutant_ind(0);  // Current index of the (looped over) mutation.
    time_t ts_t(0), te_t(0);
    double new_fit(0), new_effect(0);
    double alpha_ck(0), Delta_ck(0);

    // Compute the change in the number of cells for each strain, 
    // as well as the mutations that stem from each strain.
    for (int curr_strain = 0; curr_strain < n_strains; curr_strain++) {
        // draw the mutants
        dN = n_bac[curr_strain]*(exp(fits[curr_strain]*dt) - 1);
        
        // don't do the random draw if we have less 
        // than a single bacterium to save time
        n_mutants = (dN < 1)? 0 : gsl_ran_poisson(gsl_gen, dN*p*L);

        // the poisson rng can sometimes draw more mutants than 
        // there were in the original population; stop this.
        n_mutants = (n_mutants <= dN)? n_mutants : dN;
        n_bac[curr_strain] += (dN - n_mutants);
        nbac_tot += dN;

        time(&ts_t);
        for (int curr_mutant = 0; curr_mutant < n_mutants; curr_mutant++) {
            // draw where the mutant occurred
            mutant_ind = gsl_rng_uniform_int(gsl_gen, L);

            // store the sequence of mutations (used for testing fixed 
            // mutations in post-processing)
            auto curr_mut_order = mut_order[curr_strain];
            curr_mut_order.push_back(mutant_ind);

            // draw the new spin value and update the mutation increments.
            alpha_ck = 2*gsl_rng_uniform(gsl_gen) - 1;
            unordered_map<int, double> curr_muts = mutations_hoc[curr_strain];
            auto mut_iterator = curr_muts.find(mutant_ind);

            // Delta_ck = \alpha_k^c - \alpha_k^p
            Delta_ck = alpha_ck - alpha0s[mutant_ind] 
                - ((mut_iterator == curr_muts.cend())? 0 
                        : curr_muts[mutant_ind]);

            // curr_muts[k] = \alpha_k^P - \alpha_k^m
            curr_muts[mutant_ind] = alpha_ck - alpha0s[mutant_ind];

            // update mutation information
            mut_order.push_back(curr_mut_order);
            mutations_hoc.push_back(curr_muts);

            // compute and save the fitness for the new strain
            new_fit = compute_fitness_hoc(curr_strain, mutant_ind, 
                                          Delta_ck, 
                                          mutations_hoc[curr_strain]); 
            fits.push_back(new_fit);

            // store the fitness effect information
            auto curr_fit_effects = fit_effects[curr_strain];
            new_effect = new_fit - fits[curr_strain];
            curr_fit_effects.push_back(new_effect);
            fit_effects.push_back(curr_fit_effects);
            bin_counters[find_bin_ind(new_effect/fits[curr_strain])]++;

            // update strain counter and total number of bacteria
            new_strains++; 
            n_bac.push_back(1);
        }
    }

    time(&te_t);
    n_strains += new_strains;
}


/* Checks for strains with zero bacteria. */
void lenski_sim::check_nbac_wtf() {
    for (int ii = 0; ii < n_strains; ii++) {
        int curr_nbac = n_bac[ii];
        if (curr_nbac <= 0) { printf("WTF on strain: %d\n", ii); }
    }
}


/* Computes the fitness using the full definition, for testing other 
 * fitness computation methods. 
 * Used for +-1-valued spins. */
double lenski_sim::compute_fitness_slow(int strain_index) {
    double fit(0);
    double alpha_i, alpha_j;
    int key;
    for (int ii = 0; ii < L; ii++) {
        auto mut_it = mutations[strain_index].find(ii);
        alpha_i = (mut_it == mutations[strain_index].cend())? 
            alpha0s[ii] : -alpha0s[ii];
        fit += alpha_i*his[ii];

        if (interact) {
            for (int jj = ii+1; jj < L; jj++) {
                mut_it = mutations[strain_index].find(jj);
                alpha_j = (mut_it == mutations[strain_index].cend())? 
                    alpha0s[jj] : -alpha0s[jj];
                key = jj + L*ii;
                fit += 2*Jijs[key]*alpha_i*alpha_j;
            }
        }
    }

    return Foff + fit;
}

/* Compute the fitness using the local field approach. */
double lenski_sim::compute_fitness_lf(int parent_index,
                                      int mutation_index) {
    // find the value of alpha_kp
    auto parent_muts = mutations[parent_index];
    auto mut_it = parent_muts.find(mutation_index);
    double alpha_kp = (mut_it == parent_muts.cend())? 
        alpha0s[mutation_index] : -alpha0s[mutation_index];

    // compute the instantaneous contribution to the fitness
    double fit = fits[parent_index];
    fit -= 2*alpha_kp*his[mutation_index];

    // compute the Jalpha contribution to the fitness
    int key, row, col;
    double alpha_j(0), Jalpha(0);
    for (int jj = 0; jj < L; jj++) {
        // get alpha_jp
        mut_it = parent_muts.find(jj);
        alpha_j = (mut_it == parent_muts.cend())? 
            alpha0s[jj] : -alpha0s[jj];

        // get Jkj
        row = mutation_index <= jj? mutation_index : jj;
        col = mutation_index <= jj? jj : mutation_index;
        key = col + row*L;

        // compute the dot product
        Jalpha += Jijs[key]*alpha_j;
    }
    Jalpha *= -4*alpha_kp;

    return fit + Jalpha;
}


/* Computes the fitness using the full definition, for 
 * testing other fitness computation methods. 
 * Used for real-valued spins. */
double lenski_sim::compute_fitness_slow_hoc(int strain_index) {
    double fit(0);
    double alpha_i, alpha_j;
    int key;
    for (int ii = 0; ii < L; ii++) {
        auto mut_it = mutations_hoc[strain_index].find(ii);
        alpha_i = alpha0s[ii] 
            + ((mut_it == mutations_hoc[strain_index].cend())? 
                    0 : mut_it->second);  
        fit += alpha_i*his[ii];

        if (interact) {
            for (int jj = ii+1; jj < L; jj++) {
                mut_it = mutations_hoc[strain_index].find(jj);
                alpha_j = alpha0s[jj] 
                    + ((mut_it == mutations_hoc[strain_index].cend())? 
                            0 : mut_it->second);  
                key = jj + L*ii;
                fit += 2*Jijs[key]*alpha_i*alpha_j;
            }
        }
    }

    return Foff + fit;
}


/* Computes the fitness value for a new bacterial strain, given the parent.
 * and the identification index for the mutation. Does so for +-1-valued spins.
 * Does not require the initial strain to be all ones. */
double lenski_sim::compute_fitness(
        int parent_index, 
        int mutation_index, 
        unordered_set<int> &parent_muts, 
        double alphak_p) {

    double J_cross_muts(0);
    int first_mut, second_mut, key;

    if (interact) {
        for (const int &other_mut : parent_muts) {
            first_mut  = (other_mut < mutation_index)? 
                other_mut : mutation_index;
            second_mut = (other_mut <= mutation_index)? 
                mutation_index : other_mut;
            key = second_mut + L*first_mut;
            J_cross_muts += Jijs[key]*alpha0s[other_mut];
        }
    }

    else { J_cross_muts = 0; }

    return fits[parent_index] - alphak_p*(2*his[mutation_index] 
            + 4*Jalpha0[mutation_index] - 8*J_cross_muts);
}


/* Computes the fitness value for a new bacterial strain, given the parent.
 * and the identification index for the mutation. 
 * Does so for real-valued spins.
 * Does not require the initial strain to be all ones. */
double lenski_sim::compute_fitness_hoc(
        int parent_index, 
        int mutation_index, 
        double Delta_ck, 
        unordered_map<int, double> &parent_muts) {

    double J_cross_muts(0);
    int first_mut, second_mut, key;

    if (interact) {
        for (const auto & [other_mut, Delta_pj]: parent_muts) {
            first_mut  = (other_mut < mutation_index)? 
                other_mut : mutation_index;
            second_mut = (other_mut <= mutation_index)? 
                mutation_index: other_mut;
            key = second_mut + L*first_mut;
            J_cross_muts += Jijs[key]*Delta_pj;
        }
    }

    else { J_cross_muts = 0; }
    return fits[parent_index] + Delta_ck*(his[mutation_index] 
            + 2*(Jalpha0[mutation_index] + J_cross_muts));
}


/* Compute rank of a given strain, and store all the beneficial mutations. */
int lenski_sim::compute_rank(
        int strain_ind, 
        vector<int> &beneficial_muts, 
        double &avg_fit_inc, 
        bool store_incs) {
    // Stores the rank of this strain.
    int curr_rank(0);          

    // Computes the hypothetical fitness of a mutant 
    // strain, used to determine rank.
    double curr_new_fit(0);    

    // declare both of these guys and switch on simulation type.
    unordered_set<int> muts; unordered_map<int, double> muts_hoc;
    if (hoc) { muts_hoc = mutations_hoc[strain_ind]; }
    else { muts = mutations[strain_ind]; }
    
    // declare variables for fitness computation.
    double Delta_ck(0), Delta_pk(0), fit_inc(0), alphak_p(0);
    avg_fit_inc = 0;
    if (store_incs) { beneficial_incs.clear(); }

    // Loop over all the genes and check the fitness 
    // if a mutation were to occur there.
    for (int curr_gene = 0; curr_gene < L; curr_gene++) {
        // swap the fitness computation depending on the kind of simulation.
        if (hoc) {
            auto mut_iterator = muts_hoc.find(curr_gene);
            Delta_pk = ((mut_iterator == muts_hoc.cend())? 
                    0 : mut_iterator->second);
            Delta_ck = -2*(alpha0s[curr_gene] + Delta_pk);
            curr_new_fit = compute_fitness_hoc(strain_ind, curr_gene, 
                                               Delta_ck, muts_hoc);
        }
        else {
            auto mut_iterator = muts.find(curr_gene);

            alphak_p = (mut_iterator == muts.cend())? 
                alpha0s[curr_gene] : -alpha0s[curr_gene];

            curr_new_fit = compute_fitness(strain_ind, curr_gene, 
                                           muts, alphak_p);
        }

        // compute the fitness and store info if its a beneficial mutation.
        fit_inc = curr_new_fit - fits[strain_ind]; 
        if (fit_inc >= 0) {
            curr_rank += 1;
            avg_fit_inc += fit_inc;
            beneficial_muts.push_back(curr_gene);
            if (store_incs) { beneficial_incs.push_back(fit_inc); }
        }
    }

    avg_fit_inc /= (curr_rank > 0)? curr_rank : 1.;
    return curr_rank;
}


/* Update rank of the initial strain. */
void lenski_sim::update_rank() {
    // declare variables needed for the computation
    vector<int> beneficial_muts;
    double avg_fit_inc(0);
    int mut_ind(0);

    // compute the initial rank and keep trying to reduce the rank.
    // compute the distribution of fitness increments each time so that
    // we will have it when we hit the desired rank.
    int final_rank = compute_rank(0, beneficial_muts, avg_fit_inc, true);
    int row(0), col(0);
    while (final_rank > init_rank) {
        // draw mutation index randomly and flip the spin
        mut_ind = beneficial_muts[gsl_rng_uniform_int(gsl_gen, beneficial_muts.size())];
        alpha0s[mut_ind] *= -1;

        // apply corrections to Jalpha
        for (int i = 0; i < L; i++) {
            row = (i <= mut_ind)? i : mut_ind;
            col = (i <= mut_ind)? mut_ind : i;
            Jalpha0[i] += 2*Jijs[col + row*L]*alpha0s[mut_ind];
        }

        // recompute the rank to see where we are at.
        beneficial_muts.clear();
        final_rank = compute_rank(0, beneficial_muts, avg_fit_inc, true); 
    }

    // save initial strain information.
    ranks[0] = final_rank;
    avg_incs[0] = avg_fit_inc;
    
    // re-compute Jalpha0 and Foff for this init.
    Foff = 0; 
    Foff = 1 - (hoc? compute_fitness_slow_hoc(0) : compute_fitness_slow(0));
}

/* Run the whole simulation for n_days days with timestep dt. */
void lenski_sim::simulate_experiment(int n_days, double dt) {
    // Time the results.
    time_t dilute_start(0), dilute_end(0);
    time_t step_start(0), step_end(0);
    time_t frame_start(0), frame_end(0);

    // time values.
    double dilute_time(0),
           step_time(0),
           total_time(0),
           rank_time(0);

    int n_frames = n_days/output_interval;

    // simulate for a fixed number of generations/days.
    for (curr_day = 0; curr_day < n_days; curr_day++) {
        // keep stepping forward until we have reached the correct size.
        time(&step_start);  
        while (nbac_tot < N_f) { 
            hoc? step_forward_hoc(dt) : step_forward(dt);
        } 

        time(&step_end);
        step_time += difftime(step_end, step_start)/60.;

        // perform and time the dilution.
        time(&dilute_start); 
        dilute_gsl(); 
        time(&dilute_end);
        dilute_time += difftime(dilute_end, dilute_start)/60.;

        if (!hoc) { update_reference_strain(); }

        // measurement occurs after dilution.
        if (curr_day % output_interval == 0) { 
            output_frame_info(total_time, frame_start, frame_end, rank_time, 
                              dilute_time, step_time, curr_day/output_interval, 
                              n_frames); 
        }

    }

    // Final dump of simulation data.
    output_bac_data_bin("bac_data");
    output_mut_data_bin("mut_data");
    output_bin_counts();

    // Output total time information.
    string time_str = output_folder + "/time_info.dat";
    FILE *outf = fopen(time_str.c_str(), "w");
    fprintf(outf, "Total simulation time: %f hours.", total_time);
    fclose(outf);
}


/* Re-compute the reference strain for faster fitness computation. */
void lenski_sim::update_reference_strain() {
    // check if we actually have enough mutations that it will be
    // beneficial to re-define the reference strain
    auto max_nbac_it = std::max_element(n_bac.cbegin(), n_bac.cend());
    int dominant_index = std::distance(n_bac.cbegin(), max_nbac_it);
    unordered_set<int> dominant_mutations = mutations[dominant_index];
    int n_dominant_mutations = dominant_mutations.size();

    if (n_dominant_mutations >= reset_index * reset_fac) {
        printf("Updating reference strain.\n");
        time_t update_start(0), update_end(0);
        time(&update_start); 

        printf("n_dominant_mutations: %d, reset_index: %d, reset_fac: %d\n",
               n_dominant_mutations, reset_index, reset_fac);

        // update Jalpha0
        int key, row, col;
        for (int ii = 0; ii < L; ii++) {
            for (int jj : dominant_mutations) {
                row = (ii < jj)?  ii : jj;
                col = (ii <= jj)? jj : ii;
                key = col + row*L;
                Jalpha0[ii] -= 2*Jijs[key]*alpha0s[jj];
            }
        }

        // update alpha0
        for (int ii : dominant_mutations) { alpha0s[ii] *= -1; }

        // update each mutation set
        vector<unordered_set<int>> new_mutations;
        int strain_count = 0;
        for (auto mutation_set : mutations) {
            if (strain_count != dominant_index) {
                unordered_set<int> new_mutation_set;
                set_symmetric_difference(dominant_mutations.cbegin(),
                                         dominant_mutations.cend(),
                                         mutation_set.cbegin(),
                                         mutation_set.cend(),
                                         std::inserter(new_mutation_set,
                                                       new_mutation_set.begin()));
                new_mutations.push_back(new_mutation_set);
            }
            else { new_mutations.emplace_back(); }
            strain_count++;
        }
        mutations.swap(new_mutations);

        // change the next amount of mutations we need before resetting
        reset_index++;

        time(&update_end); 
        printf("Finished updating reference strain. Time: %g\n",
                difftime(update_end, update_start));
    }
}


void lenski_sim::output_frame_info(
        double &total_time, 
        time_t &frame_start, 
        time_t &frame_end, 
        double &rank_time,
        double dilute_time, 
        double step_time, 
        int curr_frame, 
        int n_frames) {

    // last frame ends now.
    if (curr_frame > 0) { time(&frame_end); }

    // output bacteria and mutation information.
    output_bac_data_bin("bac_data");
    time_t rank_start(0), rank_end(0);
    time(&rank_start); output_mut_data_bin("mut_data"); time(&rank_end);

    // update time information.
    double frame_time = curr_frame > 0? 
        difftime(frame_end, frame_start)/60. : 0;
    total_time += frame_time/60.;
    rank_time += difftime(rank_end, rank_start)/60.;

    // print diagnostic information.
    printf("Finished frame %d/%d. Total time: %0.3fh. \
            Frame time: %0.6fm.\nDilute time: %0.6fm. \
            Step time: %0.6fm. Rank compute time: %0.6fm.\n", 
            curr_frame, n_frames, total_time, frame_time, dilute_time, 
            step_time, rank_time);

    // output selection coefficient bin counters.
    output_bin_counts();

    // next frame starts now.
    time(&frame_start);
}


// Taken from https://stackoverflow.com/questions/1577475/c-
// sorting-and-keeping-track-of-indexes
template <typename T>
vector<size_t> sort_indices(const vector<T> &v) {
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
    return idx;
}


/* Perform the dilution step using the gsl 
 * random number generation library. */
void lenski_sim::dilute_gsl() {
    vector<double> new_nbac;
    vector<unordered_map<int, double>> new_mutations_hoc;
    vector<unordered_set<int>> new_mutations;
    vector<vector<int>> new_mut_order;
    vector<vector<double>> new_fit_effects;
    vector<double> new_fits;
    unordered_map<int, int> new_ranks;
    unordered_map<int, double> new_avg_incs;
    unordered_map<vector<int>, int, vector_hash> new_curr_strains;

    int n_choices(0), new_n_strains(0);
    double nbac_other(nbac_tot), nbac_curr(0);
    int nsamps(N_0);
    int strain(0);
    int strain_ind(0);

    // sort in descending order to minimize number of samples
    vector<size_t> sorted_indices = sort_indices(n_bac);

    while (nsamps > 0) {
        strain_ind = sorted_indices[strain];
        nbac_curr = n_bac[strain_ind];
        nbac_other -= n_bac[strain_ind];
        if (strain == n_strains-1) {  
            n_choices = std::min((int) round(nbac_curr), nsamps); 
            nsamps = 0; 
        }
        else {
            // sample from hypergeometric. slower, but exact.
            //n_choices = gsl_ran_hypergeometric(gsl_gen, 
            //                                   (int) round(nbac_curr), 
            //                                   (int) round(nbac_other), 
            //                                   nsamps); 
            
            // sample from binomial as an approximation.
            n_choices = gsl_ran_binomial_tpe(
                                gsl_gen, 
                                nbac_curr/(nbac_curr + nbac_other), 
                                nsamps); 

            n_choices = n_choices < nbac_curr? n_choices : nbac_curr;
        }
        if (n_choices > 0) {
            nsamps -= n_choices;
            new_nbac.push_back(n_choices);

            auto rank_it = ranks.find(strain_ind);
            if (rank_it != ranks.cend()) {
                new_ranks[new_n_strains] = ranks[strain_ind];
                new_avg_incs[new_n_strains] = avg_incs[strain_ind];
            }

            vector<int> mut_seq = mut_order[strain_ind];
            if (mut_seq.size() > 0) { 
                new_curr_strains[mut_seq] = new_n_strains; 
            }

            if (hoc) { new_mutations_hoc.push_back(mutations_hoc[strain_ind]); }
            else { new_mutations.push_back(mutations[strain_ind]); }
            new_mut_order.push_back(mut_order[strain_ind]);
            new_fit_effects.push_back(fit_effects[strain_ind]);
            new_fits.push_back(fits[strain_ind]);
            new_n_strains++;
        }
        strain++;
    }

    // Now that we've done the loop, update the class data structures.
    nbac_tot  = N_0;
    n_strains = new_n_strains;
    ranks.swap(new_ranks);
    avg_incs.swap(new_avg_incs);
    mutations.swap(new_mutations);
    mutations_hoc.swap(new_mutations_hoc);
    mut_order.swap(new_mut_order);
    fit_effects.swap(new_fit_effects);
    n_bac.swap(new_nbac);
    fits.swap(new_fits);
    current_strains.swap(new_curr_strains);
}


/* Outputs the number of bacteria and fitness values in binary. */
void lenski_sim::output_bac_data_bin(string file_name) {
    // dump the bacteria and fitness info
    char *bufc = new char[256];
    string output_str = output_folder + "/" + file_name + ".%d.bin";
    sprintf(bufc, output_str.c_str(), curr_day);
    FILE *outf = fopen(bufc, "wb");
    if (outf == NULL) { 
        fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                bufc, errno, strerror(errno)); 
    }
    delete [] bufc;

    float *buf = new float[2*n_strains];
    float *bp = buf;
    for (int ii = 0; ii < n_strains; ii++) { 
        *(bp++) = n_bac[ii]; *(bp++) = fits[ii]; 
    }
    fwrite(buf, sizeof(float), 2*n_strains, outf);
    delete [] buf;
    fclose(outf);

    // dump the total number of mutations up to this point
    output_str = output_folder + "/nmuts.bin";
    outf = fopen(output_str.c_str(), "ab");
    if (outf == NULL) { 
        fprintf(stderr, "Error opening nmuts file, errno %d, errstr (%s).\n", 
                errno, strerror(errno)); 
    }
    fwrite(&n_mutants_so_far, sizeof(int), 1, outf);
    fclose(outf);
}


/* Outputs mutation order, fitness increments, rank, and mean
 * fitness effect of remaining beneficial mutations in binary. */
void lenski_sim::output_mut_data_bin(string file_name) {
    // construct the mutation data output file.
    char *bufc = new char[256];
    string output_str = output_folder + "/" + file_name + ".%d.bin";
    sprintf(bufc, output_str.c_str(), curr_day);
    FILE *outf = fopen(bufc, "wb");
    if (outf == NULL) { 
        fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                bufc, errno, strerror(errno)); 
    }
    delete [] bufc;

    // declare some containers for the mutation data calculations.
    vector<int> curr_muts;
    vector<int> tmp;
    vector<double> curr_fit_effs;
    vector<float> dat;
    int curr_rank(0);
    double avg_fit_inc(0);

    // find the maximum bacteria count to get the distribution of 
    // beneficial fitness effects for the dominant strain.
    int max_nbac = *std::max_element(n_bac.cbegin(), n_bac.cend());

    // compute mutation data for each strain
    for (int ii = 0; ii < n_strains; ii++) {
        // add a separator between strains
        dat.push_back(10*L + ii + 1);

        // check if we should compute the distribution
        bool on_dominant_strain = n_bac[ii] == max_nbac;

        // add mutation locations and fitness effect data
        curr_muts = mut_order[ii];
        curr_fit_effs = fit_effects[ii];
        for (long unsigned int jj = 0; jj < curr_muts.size(); jj++) {
            dat.push_back(curr_muts[jj]);
            dat.push_back(curr_fit_effs[jj]);
        }

        // if it's time, compute the rank
        if (curr_day % rank_interval == 0) {
            
            // check if we already have rank information for this strain
            auto rank_it = ranks.find(ii);
            if ((rank_it != ranks.cend()) && !on_dominant_strain) {
                curr_rank = rank_it->second;
                avg_fit_inc = avg_incs.find(ii)->second;
            }

            // if not, compute the rank information
            else { 
                // check if we are the dominant strain - if so, we need to 
                // recompute the distribution of beneficial increments.
                curr_rank = compute_rank(ii, tmp, avg_fit_inc, on_dominant_strain); 
                ranks[ii] = curr_rank;
                avg_incs[ii] = avg_fit_inc;
            }
            
            // add rank and expected fitness increment information 
            // to the output file
            dat.push_back(curr_rank);
            dat.push_back(avg_fit_inc);
        }
        // if it's not time to output the rank, just put placeholders
        else { dat.push_back(-L); dat.push_back(-L); }
    }

    // write the output data to a file
    fwrite(&(dat[0]), sizeof(float), dat.size(), outf);
    fclose(outf);

    // write the distribution information to the file.
    if (curr_day % rank_interval == 0) {
        string fname = output_folder + "/inc_dist." 
            + std::to_string(curr_day) + ".bin";
        FILE* outf = fopen(fname.c_str(), "wb");
        if (outf == NULL) { 
            fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                    fname.c_str(), errno, strerror(errno)); 
        }
        fwrite(&(beneficial_incs[0]), sizeof(double), 
                beneficial_incs.size(), outf);
        fclose(outf);
    }
}


/* Outputs the h_i values in binary. */
void lenski_sim::output_his_bin() {
    string fname = output_folder + "/his.dat.bin";
    FILE* outf = fopen(fname.c_str(), "wb");
    if (outf == NULL) { 
        fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                fname.c_str(), errno, strerror(errno)); 
    }
    fwrite(&(his[0]), sizeof(double), L, outf);
    fclose(outf);
}


/* Outputs the Jij values to an output file in binary for L small enough. */
void lenski_sim::output_Jijs_bin() {
    if (L > 15000) { printf("L is too large to output the Jij values!\n"); }
    else {
        string fname = output_folder + "/Jijs.dat.bin";
        FILE *outf  = fopen(fname.c_str(), "wb");
        double *buf = new double[L-1];
        double *bp;
        if (outf == NULL) { 
            fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                    fname.c_str(), errno, strerror(errno)); 
        }
        for (int ii = 0; ii < L; ii++) {
            bp = buf;
            for (int jj = ii+1; jj < L; jj++) {
                int key = jj + L*ii;
                *(bp++) = Jijs[key];
            }
            fwrite(buf, sizeof(double), L-ii-1, outf);
        }
        delete [] buf;
        fclose(outf);
    }
}

/* Outputs the initial alpha_i values. */
void lenski_sim::output_alpha0s() {
    string fname = output_folder + "/alpha0s.dat";
    FILE* outf = fopen(fname.c_str(), "w");
    if (outf == NULL) { 
        fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                fname.c_str(), errno, strerror(errno)); 
    }
    for (auto curr_alf : alpha0s) { fprintf(outf, "%g\n", curr_alf); }
    fclose(outf);
}


/* Outputs the values of the row sums in binary. */
void lenski_sim::output_Jalpha_bin() {
    string fname = output_folder + "/" + "Jalpha0_bin.dat";
    FILE* outf = fopen(fname.c_str(), "wb");
    if (outf == NULL) { 
        fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                fname.c_str(), errno, strerror(errno)); 
    }
    fwrite(&(Jalpha0[0]), sizeof(double), L, outf);
    fclose(outf);
}


void lenski_sim::check_Jalpha0() {
    int key(0), row(0), col(0);
    double crow_val(0);
    for (int i = 0; i < L; i++) {
        crow_val = 0;
        for (int j = 0; j < L; j++) {
            row = (i < j)?  i : j;
            col = (i <= j)? j : i;
            key = col + row*L;
            crow_val += Jijs[key]*alpha0s[j];
        }
        assert(fabs(Jalpha0[i]-crow_val) < 1e-12);
    }
}


void lenski_sim::update_Jalpha0() {
    int key(0), row(0), col(0);
    for (int i = 0; i < L; i++) {
        Jalpha0[i] = 0;
        for (int j = 0; j < L; j++) {
            row = (i < j)?  i : j;
            col = (i <= j)? j : i;
            key = col + row*L;
            Jalpha0[i] += Jijs[key]*alpha0s[j];
        }
    }
}


/* Fills the bin_edges array with the correct number of edges based 
 * on min_select, max_select, and nbins.
 * Also ensures that deleterious and beneficial mutations 
 * can be differentiated. */
void lenski_sim::setup_bins() {
    *bin_edges = min_select;
    double ds = (max_select - min_select)/nbins;

    // ensure that the start == min_select, and the end == max_select
    for (int ii = 1; ii <= nbins; ii++) {  
        bin_edges[ii] = bin_edges[ii-1] + ds; 
    }

    // now adjust the bin edges to differentiate between 
    // beneficial and deleterious mutations.
    int flip_index = -1;
    for (int ii = 0; ii < nbins; ii++) { 
        if ((bin_edges[ii] < 0) && (bin_edges[ii+1] > 0)) { flip_index = ii; } 
    }

    // shorten the bin on the positive side, because 
    // deleterious mutations are more rare.
    if (flip_index >= 0) { bin_edges[flip_index] = 0; }

    // ensure that bin_counts is initialized
    for (int ii = 0; ii < nbins+2; ii++) { bin_counters[ii] = 0; }

}


/* Finds the index of the bin that this selection coefficient falls into. */
int lenski_sim::find_bin_ind(double select){
    if (select < min_select)      { 
        return 0;  // note min_select == bin_edges[0]
    }  
    else if (select > max_select) { 
        return nbins + 1;  // note max_select is the last element of bin_edges
    }  

    for (int ii = 1; ii <= nbins; ii++) { 
        if ((select >= bin_edges[ii-1]) && (select <= bin_edges[ii])) { 
            return ii; 
        } 
    }
    return 0;
}


void lenski_sim::output_bin_edges() {
    string fname = output_folder + "/bin_edges.dat.bin";
    FILE *outf  = fopen(fname.c_str(), "wb");
    double *bp = bin_edges;
    fwrite(bp, sizeof(double), nbins+1, outf);
    fclose(outf);
}


void lenski_sim::output_bin_counts() {
    char *bufc = new char[256];
    string fname = output_folder + "/bin_counts.%d.bin";
    sprintf(bufc, fname.c_str(), curr_day);
    FILE *outf = fopen(bufc, "wb");
    if (outf == NULL) { 
        fprintf(stderr, "Error opening file %s, errno %d, errstr (%s).\n", 
                bufc, errno, strerror(errno)); 
    }
    int *bp = bin_counters;
    fwrite(bp, sizeof(int), nbins+2, outf);
    fclose(outf);
    delete [] bufc;
}

vector<vector<double>> lenski_sim::compute_alpha0s_and_Jalpha0s(int n_inits) {
    vector<vector<double>> alpha0s_and_Jalpha0s;
    for (int ii = 0; ii < n_inits; ii++) {

        // reset alpha0s to a random initial state
        for (int jj = 0; jj < L; jj++) {
            alpha0s[jj] = (gsl_rng_uniform(gsl_gen) < 0.5)? 1 : -1;
        }

        // compute rank corresponding to this new alpha0
        update_rank();
        
        // save this result
        alpha0s_and_Jalpha0s.push_back(alpha0s);
        alpha0s_and_Jalpha0s.push_back(Jalpha0);
    }

    return alpha0s_and_Jalpha0s;
}


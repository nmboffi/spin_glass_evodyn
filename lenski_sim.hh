#ifndef LENSKI_SIM_HH
#define LENSKI_SIM_HH

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <iterator>
#include <random>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <boost/container_hash/hash.hpp>


using boost::hash_range;
using std::vector; 
using std::map;
using std::unordered_set;
using std::unordered_map;
using std::find_if;
using std::find;
using std::partial_sum;
using std::back_inserter;
using std::string;


/* Hash for vectors, allowing us to store the sequence of mutations as
 * a vector<int> and use that as a key to unordered_map. 
 * Taken from https://stackoverflow.com/questions/10405030/c- \
 * unordered-map-fail-when-used-with-a-vector-as-key. */
struct vector_hash {
    std::size_t operator()(vector<int> const& v) const {
        return hash_range(v.cbegin(), v.cend());
    }
};


/* Master class for simulation of Lenski's LTEE. */
class lenski_sim {
    public:
        /* Class constructor with sensible defaults provided by Yipei. */
        lenski_sim(
                const int _L = 4.6e8, 
                const int _N_0 = 5e6, 
                const int _N_f = 5e8,
                const double _p = 8.9e-11, 
                const double _rho = .05,
                const double _sigh = 0,   
                const double _muJ = 0, 
                const double _sigJ = 0,
                string _base_folder = "lenski_data", 
                bool _interact = true, 
                int _output_interval = 1, 
                int _init_rank = -1, 
                int _rank_interval = 1, 
                const int _nbins = 250, 
                const double _min_select = -.125,  
                const double _max_select = .125, 
                uint32_t _seed = 0,
                const bool _hoc=true, 
                const int _replicate_number = 0, 
                const bool _output_sim_info = true,
                const int _reset_fac = 15);

        /* Copy over what is needed to vary the strength of epistasis. */
        void scale_disorder(
                vector<double> unit_his, 
                vector<double> unit_Jijs);

        /* Update the initial state. */
        void copy_disorder(
                vector<double> _his, 
                vector<double> _Jijs, 
                vector<double> _alpha0s, 
                vector<double> _Jalpha0);

        /* Set up the simulation environment. */
        void allocate_space(
                uint32_t seed, 
                bool output_sim_info = true);

        /* Draw the quenched disorder. */
        void setup_disorder();

        /* Draws a Poisson random number via inversion 
         * by sequential search. */
        int draw_poisson(double lambda);

        /* Steps the simulation forward by dt seconds 
         * in the 'small' setting. */
        void step_forward(double dt);

        /* Steps the simulation forward by dt seconds in 
         * the house of cards setting. */
        void step_forward_hoc(double dt);

        /* Perform the dilution step with gsl random number generation. */
        void dilute_gsl();

        /* Run the whole simulation for n_days days with timestep dt, 
         * using curr_exp to label the output folder. */
        void simulate_experiment(
                int n_days, 
                double dt);


        /* Re-compute the reference strain for faster fitness computation. */
        void update_reference_strain();

        /* Computes the fitness value for a new bacterial strain, 
         * given the parent and the identification index for the mutation. 
         * Does so for real-valued spins. Does not require the 
         * initial strain to be all ones. */
        inline double compute_fitness_hoc(
                int parent_index, 
                int mutation_index, 
                double Delta_ck, 
                unordered_map<int, double> &parent_muts);


        /* Computes the fitness value for a new bacterial strain, 
         * given the parent and the identification index for the mutation. 
         * Does so for +-1 valued spins. Does not require the initial strain 
         * to be all ones. */
        inline double compute_fitness(
                int parent_index, 
                int mutation_index, 
                unordered_set<int> &parent_muts, 
                double alphak_p);


        /* Computes the fitness using the full definition, for testing 
         * other fitness computation methods. Used for +-1-valued spins. */
        double compute_fitness_slow(int strain_index);

        /* Compute the fitness using the local field formulation. */
        double compute_fitness_lf(int parent_index, int mutation_index);

        /* Computes the fitness using the full definition, for testing 
         * other fitness computation methods. Used for real-valued spins. */
        double compute_fitness_slow_hoc(int strain_index);

        /* Compute rank of the inital strain. */
        int compute_rank(
                int strain_ind, 
                vector<int> &beneficial_muts, 
                double &avg_fit_inc, 
                bool store_incs);

        /* Update rank of the initial strain. */
        void update_rank();

        /* Outputs bacterial information. */
        void output_bac_data_bin(string file_name);

        /* Outputs mutation and rank information. */
        void output_mut_data_bin(string file_name);

        /* Outputs the h_i values. */
        void output_his_bin();

        /* Outputs the J_{ij} values. */
        void output_Jijs_bin();

        /* Outputs the initial alpha_i values. */
        void output_alpha0s();

        /* Outputs the J*\alpha information . */
        void output_Jalpha_bin();

        /* Outputs selection coefficient binning info. */
        void output_bin_edges();
        void output_bin_counts();

        /* Wrapper function that outputs all relevant 
         * information each frame. */
        void output_frame_info(
                double &total_time, 
                time_t &frame_start, 
                time_t &frame_end, 
                double &rank_time, 
                double dilute_time, 
                double step_time, 
                int curr_frame, 
                int n_frames);

        /* Checks for a strain with 0 bacteria. */
        void check_nbac_wtf();

        /* Checks the Jalpha0 vector. */
        void check_Jalpha0();

        /* Updates the Jalpha0 vector. */
        void update_Jalpha0();

        /* Fills bin_edges with the correct values based on nbins, 
         * min_select, and max_select. */
        void setup_bins();

        /* Finds the bin index for the given selection coefficient. */
        int find_bin_ind(double select);

        /* Compute n_inits initializations for use when varying the mutation rate. */
        vector<vector<double>> compute_alpha0s_and_Jalpha0s(int n_inits);

        vector<double> get_alpha0s() { return alpha0s; }
        vector<double> get_Jalpha0() { return Jalpha0; }
        vector<double> get_his()     { return his;     }
        vector<double> get_Jijs()    { return Jijs;    }

    private:
        /* Size of the E. Coli genome. */
        const int L;

        /* Number of bacteria at the start of every day. */
        const int N_0;

        /* Number of bacteria at the end of each day. */
        const int N_f;

        /* Point mutation rate. */
        const double p;

        /* Interaction parameter. */
        const double rho;

        /* Standard deviation of the h_i distribution. */
        const double sigh;

        /* Standard deviation of the J distribution. */
        const double sigJ;

        /* Offset value for the small simulations. */
        double Foff;

        /* Current simulation day. */
        int curr_day;

        /* Current number of mutations that have occurred. */
        int n_mutants_so_far;

        /* Where to output the data. */
        string base_folder;
        string output_folder;

        /* Total current number of bacteria. */
        double nbac_tot;

        /* Total current number of strains. */
        int n_strains;

        /* Boolean indicating small simulations. */
        int sim_case;

        /* Boolean indicating whether or not the J_{ij} 
         * terms are turned on. */
        bool interact;

        /* Maps mutation indices to deviations from initial sequence.
         * Indexed such that mutations[ii] contains all the active 
         * mutations for strain ii. Used for the HOC simulation. */
        vector<unordered_map<int, double>> mutations_hoc;

        /* Stores deviations from the initial sequence. */
        vector<unordered_set<int>> mutations;

        /* Stores the orders of mutations. */
        vector<vector<int>> mut_order;

        /* Stores the fitness effect for each mutation for each strain.
         * Goes in line with mutations - i.e., 
         * mutations[strain_ind][mutation_ind] has fitness effect
         * given by fit_effects[strain_ind][mutation_ind].*/
        vector<vector<double>> fit_effects;

        /* Stores the set {\alpha_i} for the original strain.
         * The vector mutations defined above is then 
         * relative to this vector. */
        vector<double> alpha0s;

        /* Stores the rank of each strain. */
        unordered_map<int, int> ranks;

        /* Stores the distribution of beneficial fitness increments 
         * for the dominant strain. */
        vector<double> beneficial_incs;

        /* Stores the average fitness increments over the available 
         * beneficial mutations for each strain. */
        unordered_map<int, double> avg_incs;

        /* Maps a sequence of mutations to its index in other data structures.
         * Used for checking if a given strain should merge with another. */
        unordered_map<vector<int>, int, vector_hash> current_strains;

        /* Stores the number of bacteria per species. 
         * Indexed such that n_bac[ii] is the number of 
         * mutations for strain ii. */
        vector<double> n_bac;

        /* Store the fitness value for each strain. Note this only 
         * needs to be updated on the fly. 
         * Indexed such that fits[ii] is the fitness for strain ii. */
        vector<double> fits;

        /* Stores the values of h_i. 
         * Indexed such that his[i] = h_i. */
        vector<double> his;

        /* Stores the components of J*alpha0 */
        vector<double> Jalpha0;

        /* J_{ij} values mapping i+L*j -> J_{ij}. */
        //unordered_map<int, double> Jijs;
        vector<double> Jijs;

        /* Random number generator */
        gsl_rng *gsl_gen;

        /* How many days to wait before dumping data. */
        int output_interval;

        /* What the rank of the initial strain should be. */
        int init_rank;

        /* Number of days before we output the rank. */
        int rank_interval;

        /* Stores the edges of the bins for counting 
         * selection coefficients. */
        double *bin_edges;

        /* Stores the number of selection coefficients in each bin. */
        int *bin_counters;

        /* Number of bins for storing selection coefficient counts. */
        int nbins;

        /* Upper and lower bounds for the bins. */
        double min_select;
        double max_select;

        /* Whether we do discrete simulation or house of cards simulation. */
        bool hoc;

        /* Which simulation this corresponds to. */
        int replicate_number;

        /* Store the time it takes to compute the fitness. */
        double fit_time;
        double hash_time;
        double rng_time;

        /* How many times we have reset the dominant strain. */
        int reset_index;
        
        /* How many mutations before we perform the reset. */
        int reset_fac;

};

#endif

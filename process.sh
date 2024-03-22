while getopts e: flag
do
    case "${flag}" in
        e) env=${OPTARG};;
    esac
done

echo "---- Preprocess and save trajectories ----"
python scripts/preprocess_and_save_trajectories.py +params=$env
echo

echo "---- Learn dynamics from $env ----";
python scripts/create_and_save_grid.py +params=$env
echo

echo "---- Simulate $ntrajs trajectories ----";
python scripts/simulate_trajectories.py +params=$env
echo

echo "---- Plot recorded trajectories ----";
python scripts/plot_trajectories.py +params=$env ++params.trajectory_plot.trajectory_type='recorded'
echo

echo "---- Plot simulated trajectories ----";
python scripts/plot_trajectories.py +params=$env ++params.trajectory_plot.trajectory_type='simulated'
echo

echo "---- Plot probability distribution comparison ----";
python scripts/plot_histograms.py +params=$env

import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.learning import PostPre
from bindsnet.datasets import MNIST
from tqdm import tqdm


time = 500


network = Network(dt=1, learning=True)

layerIn = Input(n=28*28, traces=True)
layer1 = LIFNodes(n=100, traces=True)
layer2 = LIFNodes(n=100, traces=True)
layerOut = LIFNodes(n=10, traces=True)

con1 = Connection(source=layerIn, target=layer1, update_rule=PostPre, nu=(1e-4, 1e-2))
con2 = Connection(source=layer1, target=layer2, update_rule=PostPre, nu=(1e-4, 1e-2))
con3 = Connection(source=layer2, target=layerOut, update_rule=PostPre, nu=(1e-4, 1e-2))

outMonitor = Monitor(
    obj=layerOut,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation (if known ahead of time).
)

network.add_layer(layer=layerIn, name="inputLayer")
network.add_layer(layer=layer1, name="layer1")
network.add_layer(layer=layer2, name="layer2")
network.add_layer(layer=layerOut, name="layerOut")
#access network.layers['layer1']

network.add_connection(connection=con1, source="inputLayer", target="layer1")
network.add_connection(connection=con2, source="layer1", target="layer2")
network.add_connection(connection=con3, source="layer2", target="layerOut")
#access network.connections['layer1', 'layer2']

network.add_monitor(monitor=outMonitor, name="layerOut")
#access network.monitors[<name>].get(<state_var>)

spike_record = torch.zeros(update_interval, time, n_neurons)


'''# Create input spike data, where each spike is distributed according to Bernoulli(0.1).
input_data = torch.bernoulli(0.1 * torch.ones(time, 28*28)).byte()
inputs = {"inputLayer": input_data}

# Simulate network on input data.
network.run(inputs=inputs, time=time)

spikes = {"layerOut": outMonitor.get("s")}
voltages = {"B": outMonitor.get("v")}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()
'''



dataset = MNIST(
    PoissonEncoder(time=time, dt=1),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128)]
    ),
)



dataloader = torch.utils.data.DataLoader(
    dataset, batch_size = 1, shuffle = True
)

for step, batch in enumerate(tqdm(dataloader)):
    inputs = {"inputLayer": batch["encoded_image"].view(time,28*28)}
    
    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"].append(
        100
        * torch.sum(label_tensor.long() == all_activity_pred).item()
        / len(label_tensor)
    )
    accuracy["proportion"].append(
        100
        * torch.sum(label_tensor.long() == proportion_pred).item()
        / len(label_tensor)
    )

    print(
        "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
        % (
            accuracy["all"][-1],
            np.mean(accuracy["all"]),
            np.max(accuracy["all"]),
        )
    )
    print(
        "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
        % (
            accuracy["proportion"][-1],
            np.mean(accuracy["proportion"]),
            np.max(accuracy["proportion"]),
        )
    )

    # Assign labels to excitatory layer neurons.
    assignments, proportions, rates = assign_labels(
        spikes=spike_record,
        labels=label_tensor,
        n_labels=n_classes,
        rates=rates,
    )

    labels = []

    labels = batch["label"]
    network.run(inputs=inputs, time=time, input_time_dim=1)


#inputData = torch.bernoulli(0.1 * torch.ones(time, layerIn.n)).byte()
-- changing to LIF_nodes does not seem to work

-- moving seems to give fine results:
```python 
 exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=0.0,
            reset=5.0,
            thresh=12.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=0.0,
            reset=15.0,
            thresh=20.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )
```

-- multiply by 1000 does not work
- thresh
- w

-- post trains scale works
```python
class ShowCaseNet(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=0.0,
            reset=5.0,
            thresh=12.0,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=0.0,
            reset=15.0,
            thresh=20.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")
        .....
        

if mod:
    network_old.Ae.thresh *= 1000
    network_old.Ae.learning = False

    network_old.Ae_to_Ai.w *= 1000
    network_old.Ae_to_Ai.wmax *= 1000
    network_old.Ae_to_Ai.training = False

    network_old.Ai.thresh *= 1000
    network_old.Ai.learning = False

    network_old.Ai_to_Ae.w *= 1000
    network_old.Ai_to_Ae.wmin *= 1000
    network_old.Ai_to_Ae.training = False

    network_old.X.learning = False

    network_old.X_to_Ae.w *= 1000
    network_old.X_to_Ae.wmax *= 1000
    network_old.X_to_Ae.training = False
    network_old.X_to_Ae.norm *= 1000
```

prescale: accuacy: 0.505
post scale:  accuacy: 0.57  (interesting)

### proper hyper parameter chance works:

```python
network = ShowCaseNet(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc*1000,
    inh=inh*1000,
    dt=dt,
    norm=78.4*1000,
    nu=[0, 1e-2],
    inpt_shape=(1, 28, 28),
)

class ShowCaseNet(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1000.0,
        norm: float = 78.4,
        theta_plus: float = 0.05*1000,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=0.0,
            reset=5.0,
            thresh=12.0*1000,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=0.0,
            reset=15.0,
            thresh=20.0*1000,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3*1000 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")
```
Accuacy of 200 test with 2000 training 0.665


### converting to lif

to 0 DiehlAndCookNodes train 2000 scalled by 1000 then converted to all LIF acc: 0.43 loss of 10%
## trained scaled DiehlAndCookNetwork
200 test accuacy: 0.765

### converted to LIF 

``` python
def toLIF(network : Network):
    new_network = Network(dt=1, learning=True)
    input_layer = Input(
        n=network.X.n, shape=network.X.shape, traces=True, tc_trace=network.X.tc_trace.item()
    )
    exc_layer = LIFNodes(
        n=network.Ae.n,
        traces=True,
        rest=network.Ai.rest.item(),
        reset=network.Ai.reset.item(),
        thresh=network.Ai.thresh.item(),
        refrac=network.Ai.refrac.item(),
        tc_decay=network.Ai.tc_decay.item(),
    )
    inh_layer = LIFNodes(
        n=network.Ai.n,
        traces=False,
        rest=network.Ai.rest.item(),
        reset=network.Ai.reset.item(),
        thresh=network.Ai.thresh.item(),
        tc_decay=network.Ai.tc_decay.item(),
        refrac=network.Ai.refrac.item(),
    )

    # Connections
    w = network.X_to_Ae.w
    input_exc_conn = Connection(
        source=input_layer,
        target=exc_layer,
        w=w,
        update_rule=PostPre,
        nu=network.X_to_Ae.nu,
        reduction=network.X_to_Ae.reduction,
        wmin=network.X_to_Ae.wmin,
        wmax=network.X_to_Ae.wmax,
        norm=network.X_to_Ae.norm,
    )
    w = network.Ae_to_Ai.w
    exc_inh_conn = Connection(
        source=exc_layer, target=inh_layer, w=w, wmin=network.Ae_to_Ai.wmin, wmax=network.Ae_to_Ai.wmax
    )
    w = network.Ai_to_Ae.w
    
    inh_exc_conn = Connection(
        source=inh_layer, target=exc_layer, w=w, wmin=network.Ai_to_Ae.wmin, wmax=network.Ai_to_Ae.wmax
    )

    # Add to network
    new_network.add_layer(input_layer, name="X")
    new_network.add_layer(exc_layer, name="Ae")
    new_network.add_layer(inh_layer, name="Ai")
    new_network.add_connection(input_exc_conn, source="X", target="Ae")
    new_network.add_connection(exc_inh_conn, source="Ae", target="Ai")
    new_network.add_connection(inh_exc_conn, source="Ai", target="Ae")

    exc_voltage_monitor = Monitor(new_network.layers["Ae"], ["v"], time=500)
    inh_voltage_monitor = Monitor(new_network.layers["Ai"], ["v"], time=500)
    new_network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    new_network.add_monitor(inh_voltage_monitor, name="inh_voltage")

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(new_network.layers[layer], state_vars=["s"], time=time)
        new_network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    return new_network
```
Gives accuacy 0.62 loss of 13%   
- norm scales:
0.9 : 0.575  
0.95 : 0.58  
1.05 : 0.62   
1.1 : 0.63  
1.2 : 0.645
1.5 : 0.665
2.0 : 0.665
2.4 : 0.68
2.45 : 0.634 long
2.5 : 0.7 0.634 long
2.55 : 0.621
2.6 : 0.665
3.0 : 0.64
4.0 : 0.685
5.0 : 0.655
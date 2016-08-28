package utils.yield;

import java.util.Optional;


public interface Yielderable<T> extends Iterable<T> {

    void execute(YieldDefinition<T> builder);

    default ClosableIterator<T> iterator() {
        YieldDefinition<T> yieldDefinition = new YieldDefinition<>();
        Thread collectorThread = new Thread(() -> {
            yieldDefinition.waitUntilFirstValueRequested();
            try {
                execute(yieldDefinition);
            } catch (BreakException e) {/*ignored*/}
            yieldDefinition.signalComplete();
        });
        collectorThread.setDaemon(true);
        collectorThread.start();
        yieldDefinition.onClose(collectorThread::interrupt);
        return yieldDefinition.iterator();
    }
}

interface Message<T> {
    Optional<T> value();
    static <T> Message<T> message(T value) {
        return () -> Optional.of(value);
    }
}

interface Completed<T> extends Message<T> {
    static <T> Completed<T> completed() { return Optional::empty; }
}

interface FlowControl {
    FlowControl youMayProceed = new FlowControl() {};
}

class BreakException extends RuntimeException {
    public synchronized Throwable fillInStackTrace() {
        return null;
    }
}

interface Then<T> {
    void then(Runnable r);
}
class IfAbsent {
    public static <T> Then<T> ifAbsent(Optional<T> optional) {
        return runnable -> {
            if (!optional.isPresent()) runnable.run();
        };
    }
}